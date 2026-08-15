"""ppo_loop.py —— 自研 PPO 训练循环（模块② VLA-as-policy 融合，M2 实现）。

申请书承诺"以 OpenVLA 为基础 + PPO 构建 StARe-PPO"。rsl_rl 的 OnPolicyRunner 期望
MLP 风格 actor_critic，与 VLA 策略形态不匹配，因此本模块实现**轻量自研 PPO 循环**：

    VLA 策略(冻结视觉塔 + 可训头) → act → ActionOutputInterface → env.step
      → 阶段感知奖励 + 阶段信号 → StageFeedback 回灌 → 存入 buffer
      → GAE 优势估计 → PPO clip 更新（只更新头，视觉特征预提取）

关键优化：**视觉特征预提取**——每个 rollout 只跑一次冻结视觉塔前向，把特征 ``x``
（含阶段反馈）存进 buffer，PPO 多轮 minibatch 更新只回传可训练的头，省显存/耗时。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.config import Settings
from ..core.logging import get_logger
from .action_interface import IKRelActionInterface
from .online_feedback import StageFeedback
from .vla_policy import VisionFeatureExtractor, VisionOnlyPPOPolicy

logger = get_logger(__name__)


def _zero_feedback(B: int, dim: int) -> torch.Tensor:
    return torch.zeros(B, dim, dtype=torch.float32)   # 头在 CPU，反馈用 CPU 张量


def _build_policy(settings: Settings) -> VisionOnlyPPOPolicy:
    fe = VisionFeatureExtractor(settings, device="cuda")
    return VisionOnlyPPOPolicy(
        features_dim=fe.features_dim,
        action_dim=7,
        feature_extractor=fe,
        stage_feedback_dim=len(settings.stages),
    )


def _collect_rollout(env, policy, iface, feedback, stages, horizon: int) -> dict:
    """收集 ``horizon`` 步 rollout，预提取特征存入 buffer。

    可训头在 CPU（M2：GPU backward 与 Isaac 渲染冲突），因此 buffer 全在 CPU；
    env 动作移回 GPU 再 step。

    Returns:
        {"x": [T,B,F], "action": [T,B,7], "old_log_prob": [T,B],
         "reward": [T,B], "value": [T,B+1], "done": [T,B], "stage": [T,B]}（全 CPU）
    """
    from ..stages.rewards_isaac import detect_stage_from_env

    env_dev = env.unwrapped.device
    buf: dict[str, list[torch.Tensor]] = {
        "x": [], "action": [], "old_log_prob": [], "reward": [], "value": [], "done": [], "stage": [],
    }

    obs, _ = env.reset()
    images = obs["policy"]["table_cam"]           # [B,H,W,3] uint8（VisionFeatureExtractor 内部转 numpy）
    fb = feedback.feedback()
    if fb is None:
        fb = _zero_feedback(images.shape[0], len(stages))
    x = policy.features(images, fb)               # CPU（头在 CPU）
    torch.cuda.synchronize()   # 视觉塔 GPU 前向与 Isaac 同步，防竞态

    for _ in range(horizon):
        action, log_prob, value = policy.sample_from_x(x)
        env_action = iface.to_env_action(action).to(env_dev)   # CPU → GPU 喂给 env
        obs, reward, terminated, truncated, _ = env.step(env_action)

        done = (terminated | truncated).float()
        stage = detect_stage_from_env(env.unwrapped)
        feedback.on_step(None, None, None, stage, done)

        # 目标值 detach（PPO 标准：不带着收集时的计算图，否则更新 backward 冲突）
        buf["x"].append(x)
        buf["action"].append(action.detach())
        buf["old_log_prob"].append(log_prob.detach())
        buf["reward"].append(reward.cpu())
        buf["value"].append(value.detach())
        buf["done"].append(done.cpu())
        buf["stage"].append(stage.cpu())

        # 下一步特征（含最新阶段反馈）
        images = obs["policy"]["table_cam"]
        fb = feedback.feedback()
        if fb is None:
            fb = _zero_feedback(images.shape[0], len(stages))
        x = policy.features(images, fb)
        torch.cuda.synchronize()

    # 最后一步的 next-value（bootstrap，detach 作目标）
    buf["value"].append(policy.value_x(x).detach())
    return {k: torch.stack(v, dim=0) for k, v in buf.items()}  # [T+1/B, ...]


def _compute_gae(rollout: dict, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE 优势估计 + 回报。"""
    reward = rollout["reward"]       # [T,B]
    done = rollout["done"]           # [T,B]
    value = rollout["value"]         # [T+1,B]
    T = reward.shape[0]
    advantage = torch.zeros_like(reward)
    gae = torch.zeros_like(reward[0])
    for t in reversed(range(T)):
        next_nonterm = 1.0 - done[t]
        delta = reward[t] + gamma * value[t + 1] * next_nonterm - value[t]
        gae = delta + gamma * lam * next_nonterm * gae
        advantage[t] = gae
    returns = advantage + value[:-1]
    return advantage, returns


def _flat_feat(t: torch.Tensor) -> torch.Tensor:
    """特征/动作张量：``[T,B,F]`` → ``[T*B,F]``；已是 2D（测试）原样。"""
    return t.flatten(0, 1) if t.dim() >= 3 else t


def _flat_scalar(t: torch.Tensor) -> torch.Tensor:
    """每步标量张量：``[T,B]`` → ``[T*B]``；已是 1D（测试）原样。"""
    return t.flatten(0) if t.dim() >= 2 else t


def _ppo_update(policy, optimizer, rollout, advantage, returns, ppo_epochs, mini_batches, clip) -> dict:
    """PPO clip 更新（只更新可训练头，x 已预提取）。"""
    x = _flat_feat(rollout["x"])                   # [N, F]
    action = _flat_feat(rollout["action"])         # [N, 7]
    old_log_prob = _flat_scalar(rollout["old_log_prob"])
    adv = _flat_scalar(advantage)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = _flat_scalar(returns)

    mb_size = max(1, x.shape[0] // mini_batches)
    for _ in range(ppo_epochs):
        perm = torch.randperm(x.shape[0])
        for i in range(0, x.shape[0], mb_size):
            idx = perm[i:i + mb_size]
            x_mb, a_mb, old_mb, adv_mb, ret_mb = x[idx], action[idx], old_log_prob[idx], adv[idx], ret[idx]
            log_prob, entropy, value = policy.evaluate_x(x_mb, a_mb)
            ratio = (log_prob - old_mb).exp()
            surr1 = ratio * adv_mb
            surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret_mb)
            entropy_loss = -entropy.mean()
            total = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.mean().item()}


def train_vla_in_loop(
    settings: Settings,
    instruction: str | None = None,
    *,
    num_envs: int = 16,
    max_iterations: int = 50,
    headless: bool = True,
    horizon: int = 24,
    ppo_epochs: int = 5,
    mini_batches: int = 4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    lr: float = 1e-3,
    cam_res: int = 128,
    device: str = "cuda",
) -> VisionOnlyPPOPolicy:
    """VLA-as-policy 融合训练（自研 PPO 循环，**需 Isaac 环境**）。

    语义分离驱动 → 视觉环境（id_visuomotor + 阶段奖励）→ 冻结视觉塔策略 → 自研 PPO。
    """
    from ..envs.cfg_surgery import build_vision_env_cfg
    from ..stages.semantic import SemanticSeparator, plan_targets
    from .runner import create_raw_env

    instruction = instruction or settings.task["desc"]
    plan = SemanticSeparator().parse(instruction)
    cube_grasp, cube_stack, active = plan_targets(
        plan,
        default_grasp=settings.task["cube_to_grasp"],
        default_stack=settings.task["cube_to_stack_on"],
    )
    stages = list(settings.stages)

    env_cfg = build_vision_env_cfg(
        settings, num_envs, settings.ppo["seed"],
        cube_to_grasp=cube_grasp, cube_to_stack_on=cube_stack,
        active_stages=active, cam_res=cam_res,
    )
    env, sim_app = create_raw_env(
        settings.task["id_visuomotor"], env_cfg, headless=headless, enable_cameras=True,
    )

    policy = _build_policy(settings)   # 头在 CPU
    iface = IKRelActionInterface()
    feedback = StageFeedback(stages)
    optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=lr)   # CPU 参数 → CPU 优化器

    print(f"[train_vla] 融合训练启动：{num_envs} env × {max_iterations} iter | 指令={instruction!r} "
          f"活动阶段={active} 显存预算 {torch.cuda.memory_allocated()/1e9:.2f}GB（头在 CPU）")
    for it in range(max_iterations):
        rollout = _collect_rollout(env, policy, iface, feedback, stages, horizon)
        advantage, returns = _compute_gae(rollout, gamma, lam)
        losses = _ppo_update(policy, optimizer, rollout, advantage, returns, ppo_epochs, mini_batches, clip)

        mean_rew = rollout["reward"].mean().item()
        # 阶段跨越次数（相邻步阶段上升的比例）与成功终止
        stage = rollout["stage"]
        transitions = (stage[1:] > stage[:-1]).float().mean().item()
        if it % 5 == 0 or it == max_iterations - 1:
            print(f"iter {it:>3d} | reward {mean_rew:6.2f} | trans {transitions:.2f} | "
                  f"pl {losses['policy_loss']:.3f} vl {losses['value_loss']:.3f} | "
                  f"显存 {torch.cuda.memory_allocated()/1e9:.2f}GB")

    env.close()
    sim_app.close()
    return policy
