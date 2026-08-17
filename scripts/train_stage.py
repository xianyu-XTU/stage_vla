"""train_stage.py —— 分阶段 RL：抓/抬/移动/堆叠 分开训练（**需 Isaac 环境**）。

把长程任务拆成 4 个独立的 RL 问题，每阶段单独训练一个策略（课程式）：
    --stage grasp  奖励=抓取到方块
    --stage lift   奖励=方块被抬起（官方机制）
    --stage move   奖励=方块靠近目标
    --stage stack  奖励=堆叠成功
每阶段独立 checkpoint 存 logs/stage_<name>/。

用法::

    python tools\\run_isaaclab.py scripts\\train_stage.py --stage lift --max_iterations 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.rl.runner import train  # noqa: E402

# 各阶段的奖励配置（奖励函数 + 权重）
STAGE_REWARDS = {
    "grasp": ("object_grasp_combined_reward", 5.0),   # 对侧定位 + 显式闭合（社区完整方案）
    "lift": ("object_is_lifted_reward", 15.0),
    "move": ("object_near_goal_reward", 5.0),
    "stack": ("object_stacked_dense_reward", 10.0),   # 稠密奖励（稀疏的学不会）
}


def build_stage_env_cfg(settings, stage: str, num_envs: int, seed: int):
    """按阶段构造环境：只给该阶段完成奖励（+接近塑形引导）。"""
    from isaaclab.envs import mdp
    from isaaclab.managers import RewardTermCfg as RewardTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils.configclass import configclass
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from stage_vla.stages.rewards_isaac import (
        drop_penalty_reward,
        object_grasp_combined_reward,
        object_is_lifted_reward,
        object_near_goal_reward,
        object_stacked_dense_reward,
        red_on_blue_success,
        stage_potential_reward,
    )

    env_cfg = load_cfg_from_registry(settings.task["id_state"], "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.observations.policy.concatenate_terms = True

    func_name, weight = STAGE_REWARDS[stage]
    stage_func = {
        "object_grasp_combined_reward": object_grasp_combined_reward,
        "object_is_lifted_reward": object_is_lifted_reward,
        "object_near_goal_reward": object_near_goal_reward,
        "object_stacked_dense_reward": object_stacked_dense_reward,
    }[func_name]

    @configclass
    class StageRewardsCfg:
        action_penalty = RewardTerm(func=mdp.action_l2, weight=settings.reward_weights["action_penalty"])
        # 回归修复：stage_potential_reward 内部已乘 progress_shaping，weight 必须是 1.0，
        # 否则 3.0×3.0=9.0 重复放大（交接文档 8 节）
        stage_progress = RewardTerm(
            func=stage_potential_reward,
            params={"weights": settings.reward_weights, "gamma": 0.99},
            weight=1.0,
        )
        stage_goal = RewardTerm(func=stage_func, weight=weight)
        # 所有阶段都保持抓稳（组合抓取信号）——接手后不丢抓取
        stage_grasp_hold = RewardTerm(func=object_grasp_combined_reward, weight=3.0)
        drop_penalty = RewardTerm(
            func=drop_penalty_reward,
            weight=settings.reward_weights.get("drop_penalty", 10.0),
        )

    env_cfg.rewards = StageRewardsCfg()
    # 项目自定 success（红块在蓝块上）替代官方 cubes_stacked 三块塔（叠放方向相反，
    # 导致 success 恒 0）——分阶段训练也按真实任务目标评价
    env_cfg.terminations.success = DoneTerm(func=red_on_blue_success)
    return env_cfg


def _wire_state_bank(env_cfg, args) -> object | None:
    """接线阶段成功状态库课程（交接文档 11 节，默认不启用）。

    - **采集**：``rewards_isaac._grasp_state`` 在稳定抓取首次达标时置
      ``env.extras["_stage_grasp_just_stable"]``；``capture_stage_states`` 事件
      （mode="env_step"）每步廉价检查，命中才把该环境完整状态存入 bank；
    - **重置**：``reset_from_stage_bank`` 事件（mode="reset"）以 ``--bank_ratio``
      概率从 bank 采样并写回 sim（80/20 混合课程），其余默认初始化。

    返回 bank；未启用（无 --state_bank / --bank_from）返回 None。
    注意：此机制需 Isaac 环境验证后再开长训（交接文档 11 节 stage reset 根因）。
    """
    if not (args.state_bank or args.bank_from):
        return None
    from isaaclab.managers import EventTermCfg as EventTerm

    from stage_vla.rl.state_bank import StageStateBank, capture_stage_states, reset_from_stage_bank

    bank = StageStateBank()
    if args.bank_from:
        bank_file = Path(args.bank_from)
        if not bank_file.is_file():
            raise FileNotFoundError(f"bank_from 不存在：{bank_file}")
        bank.load(bank_file)
        print(f"[train_stage] 已加载状态库 {bank_file}（{bank.count()} 条）", flush=True)

    if getattr(env_cfg, "events", None) is None:
        raise RuntimeError("state_bank 需要 env_cfg.events（任务默认事件），当前为空")

    if args.state_bank:
        env_cfg.events.capture_stage_states = EventTerm(
            func=capture_stage_states,
            params={"bank": bank, "stage": "grasp"},
            mode="env_step",
        )
    if args.bank_from and bank.count(args.stage) > 0:
        env_cfg.events.state_bank_reset = EventTerm(
            func=reset_from_stage_bank,
            params={"bank": bank, "stage": args.stage, "ratio": args.bank_ratio},
            mode="reset",
        )
    return bank


def main() -> int:
    parser = argparse.ArgumentParser(description="分阶段 RL 训练")
    parser.add_argument("--stage", choices=list(STAGE_REWARDS), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="渐进微调起点（上阶段 checkpoint，策略继承），如 logs/stage_grasp/model_2000.pt")
    parser.add_argument("--state_bank", action="store_true",
                        help="启用阶段成功状态库课程：本阶段训练时采集成功状态并写回 bank 文件")
    parser.add_argument("--bank_from", type=str, default=None,
                        help="从上一阶段保存的 bank 加载（如 logs/stage_grasp/grasp_bank.pt），"
                             "以 ratio 概率从该库采样做 reset 初始化")
    parser.add_argument("--bank_ratio", type=float, default=0.8, help="reset 时从 bank 采样初始化的概率")
    args = parser.parse_args()

    settings = load_settings()
    env_cfg = build_stage_env_cfg(settings, args.stage, args.num_envs, settings.ppo["seed"])

    from stage_vla.rl.runner import build_agent_cfg

    agent_cfg = build_agent_cfg(dict(settings.ppo))
    log_root = _ROOT / "logs" / f"stage_{args.stage}"
    log_root.mkdir(parents=True, exist_ok=True)

    bank = _wire_state_bank(env_cfg, args)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else None

    print(f"[train_stage] 阶段={args.stage} 奖励={STAGE_REWARDS[args.stage]} "
          f"{args.num_envs}env×{args.max_iterations}iter "
          f"init_ckpt={init_ckpt.name if init_ckpt else '无'} "
          f"state_bank={'开' if args.state_bank or args.bank_from else '关'}", flush=True)
    train(
        settings.task["id_state"], env_cfg, agent_cfg,
        args.max_iterations, headless=args.headless, log_root=log_root,
        init_ckpt=init_ckpt,
    )
    print(f"[train_stage] 完成，checkpoint 在 {log_root}", flush=True)
    if args.state_bank and bank is not None:
        bank_file = log_root / f"stage_{args.stage}_bank.pt"
        bank.save(bank_file)
        print(f"[train_stage] 已保存状态库 {bank_file}（{bank.count()} 条）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
