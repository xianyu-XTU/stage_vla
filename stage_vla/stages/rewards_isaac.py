"""rewards_isaac.py —— Isaac Lab RewardTerm 适配层（**需要 Isaac 环境**）。

将 :mod:`stage_vla.stages.rewards` 的纯函数包装成 Isaac Lab ``RewardTerm(func=..., weight=...)``
可接受的 ``func(env, **params) -> [N]`` 接口，并做两处关键校正：

- **首帧处理**：reward 在 reset 前计算，用 ``env.episode_length_buf == 1`` 识别新回合首帧，
  把缓存的 prev 对齐 cur 并置零该帧，杜绝"刚 reset 首帧虚假势能差"。
- **dt 校正**：reward manager 会把每项 ``* dt``；本模块返回 ``值 / env.step_dt`` 抵消，
  使 config 权重即为每步名义值（可读、可调）。

注意：本文件 import Isaac Lab（``isaaclab`` / ``isaaclab_tasks``），只能通过
``isaaclab.bat`` 提供的 Python 环境运行；纯张量部分见 ``rewards.py``。
"""

from __future__ import annotations

import torch

from . import calculator
from .detector import StageDetector
from .rewards import (
    first_frame_mask,
    potential_shaping,
    stage_completion_reward,
    stage_completion_reward_first_time,
)

# 模块级缓存的阶段检测器（避免每个 reward 调用重建）
_detector: StageDetector | None = None
# 活动阶段（语义计划驱动：指令未覆盖的阶段不参与势能塑形）
_active_stages: list[str] | None = None


def _get_detector(stages: list[str] | None = None, thresholds: dict | None = None) -> StageDetector:
    global _detector
    if _detector is None:
        _detector = StageDetector(stages=stages, thresholds=thresholds or {})
    return _detector


def _cube_positions_from_env(env, cube_to_grasp: str, cube_to_stack_on: str) -> dict[str, torch.Tensor]:
    """从 Isaac 环境读取各方块位置。

    **回归修复**：``root_pos_w`` 已是世界坐标（与 ``ee_frame.target_pos_w`` 同一坐标系），
    旧代码再加 ``env_origins`` 属重复偏移，导致 ee 到方块的接近距离被算成 ~1.9m（实际 ~0.12m），
    阶段进度恒为 0、阶段感知奖励静默失效。
    """
    scene = env.scene
    return {
        name: scene[name].data.root_pos_w
        for name in (cube_to_grasp, cube_to_stack_on, "cube_3")
    }


def _grasp_stacked_signals(env, cube_to_grasp: str, cube_to_stack_on: str) -> tuple[torch.Tensor, torch.Tensor]:
    """读取抓取 / 堆叠布尔信号（复用任务 mdp 的函数）。"""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    grasp = mdp.object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(cube_to_grasp),
    )
    stacked = mdp.object_stacked(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        upper_object_cfg=SceneEntityCfg(cube_to_grasp),
        lower_object_cfg=SceneEntityCfg(cube_to_stack_on),
    )
    return grasp.float(), stacked.float()


def _stage_signals_from_env(env) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """统一读取阶段判定所需信号：末端位置 / 方块位置 / 抓取与堆叠信号。"""
    # 兼容 RslRlVecEnvWrapper（.scene 在内层 ManagerBasedRLEnv 上）
    env = getattr(env, "unwrapped", env)
    scene = env.scene
    ee_w = scene["ee_frame"].data.target_pos_w[:, 0, :]
    cube_to_grasp = _get_detector().cube_to_grasp
    cube_to_stack_on = _get_detector().cube_to_stack_on
    cube_pos = _cube_positions_from_env(env, cube_to_grasp, cube_to_stack_on)
    grasp, stacked = _grasp_stacked_signals(env, cube_to_grasp, cube_to_stack_on)
    return ee_w, cube_pos, grasp, stacked


def detect_stage_from_env(env) -> torch.Tensor:
    """从 Isaac 环境检测当前阶段索引 ``[N]``（PPO 循环 / StageFeedback 用）。"""
    det = _get_detector()
    ee_w, cube_pos, grasp, stacked = _stage_signals_from_env(env)
    return calculator.signals_to_stage(
        ee_w, cube_pos, grasp, stacked,
        stages=det.stages, thresholds=det.thresholds,
        cube_to_grasp=det.cube_to_grasp, cube_to_stack_on=det.cube_to_stack_on,
    )


def compute_stage_progress(env) -> torch.Tensor:
    """计算当前各阶段完成度 ``[N, n_stages]``（Isaac Lab 环境）。"""
    ee_w, cube_pos, grasp, stacked = _stage_signals_from_env(env)
    det = _get_detector()
    return calculator.signals_to_progress(
        ee_w, cube_pos, grasp, stacked,
        stages=det.stages, thresholds=det.thresholds,
        cube_to_grasp=det.cube_to_grasp, cube_to_stack_on=det.cube_to_stack_on,
    )


def stage_potential_reward(env, weights: dict | None = None, gamma: float = 0.99) -> torch.Tensor:
    """阶段势能塑形奖励（Isaac Lab RewardTerm，func(env, **params) -> [N]）。

    ``r = (γ·φ(s_t) − φ(s_{t−1})) * progress_shaping_weight``，首帧置零。
    φ 用非饱和距离势能（:func:`_shaping_potential`），消除随机游走负偏置。
    """
    cur = _shaping_potential(env)              # [N]
    first = first_frame_mask(env.episode_length_buf)

    prev = env.extras.get("_stage_prev_potential")
    prev = torch.zeros_like(cur) if prev is None else prev.to(cur.device)
    prev = torch.where(first, cur, prev)       # 首帧对齐

    env.extras["_stage_prev_potential"] = cur.clone()

    delta = gamma * cur - prev                 # γφ_t − φ_{t−1}
    delta = torch.where(first, torch.zeros_like(delta), delta)  # 首帧不给塑形
    scale = (weights or {}).get("progress_shaping", 1.0)
    return (delta * scale) / env.step_dt


def stage_transition_reward(env, weights: dict | None = None) -> torch.Tensor:
    """阶段完成奖励（Isaac Lab RewardTerm，func(env, **params) -> [N]）。首帧置零。

    **防奖励黑客**：每个阶段每 episode 只奖励首次进入（用 ``_stage_done`` 掩码），
    杜绝"反复跨阶段刷完成奖"（M2 长训练诊断：stage_transition 峰值 21 而 success=0）。
    """
    det = _get_detector()
    ee_w, cube_pos, grasp, stacked = _stage_signals_from_env(env)
    cur_stage = calculator.signals_to_stage(
        ee_w, cube_pos, grasp, stacked,
        stages=det.stages, thresholds=det.thresholds,
        cube_to_grasp=det.cube_to_grasp, cube_to_stack_on=det.cube_to_stack_on,
    )
    first = first_frame_mask(env.episode_length_buf)

    prev = env.extras.get("_stage_prev_stage")
    prev = cur_stage.clone() if prev is None else prev.to(cur_stage.device)
    prev = torch.where(first, cur_stage, prev)  # 首帧对齐

    env.extras["_stage_prev_stage"] = cur_stage.clone()

    # 已奖励阶段掩码（reset 首帧清零）
    done = env.extras.get("_stage_done")
    if done is None:
        done = torch.zeros(cur_stage.shape[0], len(det.stages), dtype=torch.bool, device=cur_stage.device)
    else:
        done = done.to(cur_stage.device)
    done = torch.where(first.unsqueeze(1), torch.zeros_like(done), done)

    bonus, new_done = stage_completion_reward_first_time(cur_stage, prev, done, weights or {}, det.stages)
    env.extras["_stage_done"] = new_done

    bonus = torch.where(first, torch.zeros_like(bonus), bonus)
    return bonus / env.step_dt


def object_is_lifted_reward(env, minimal_height: float = 0.04) -> torch.Tensor:
    """方块被抬起奖励（官方 Isaac-Lift 机制）：目标方块 root z > 阈值 → 1.0。

    **关键**：这是驱动策略真正"抓起并抬走"方块的核心稠密信号（官方 weight 15）。
    我们的阶段奖励原本没有这个"方块实际被抬起"的项——策略只学跨阶段信号，
    从没被要求真的把方块抬离桌面。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    return torch.where(cube[:, 2] > minimal_height, 1.0, 0.0)


def object_grasped_reward(env) -> torch.Tensor:
    """抓取奖励（分阶段 RL 用）：目标方块被抓取 → 1.0。"""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    return mdp.object_grasped(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(det.cube_to_grasp),
    ).float()


def object_grasped_opposite_reward(env) -> torch.Tensor:
    """**对侧抓取奖励**（社区方案，Isaac Lab Issue #204）：奖励手指在方块两侧。

    朴素距离奖励产生"手指同侧"局部最优（抓不稳、掉落）。本函数验证两根手指
    指向方块的向量方向：方块在手指之间时向量方向相反（点积小）→ 奖励高；
    手指同侧时点积大 → 奖励低。结合手指距离，同时奖励"靠近 + 对侧"。

    .. code-block::

        vec_l = lfinger - cube;  vec_r = rfinger - cube
        reward = 1 - tanh(sum(vec_l * vec_r)) * (|vec_l| + |vec_r|)
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    lf_id = body_names.index("panda_leftfinger")
    rf_id = body_names.index("panda_rightfinger")
    lfinger = robot.data.body_pos_w[:, lf_id]
    rfinger = robot.data.body_pos_w[:, rf_id]
    cube = env.scene[det.cube_to_grasp].data.root_pos_w

    vec_l = lfinger - cube
    vec_r = rfinger - cube
    direction = (vec_l * vec_r).sum(-1)          # 点积：对侧为负 → tanh 负 → 奖励高
    l_dist = vec_l.norm(dim=-1)
    r_dist = vec_r.norm(dim=-1)
    return 1.0 - torch.tanh(direction) * (l_dist + r_dist)


def object_near_goal_reward(env, std: float = 0.1, object_cfg: str | None = None) -> torch.Tensor:
    """移动奖励（分阶段 RL 用）：目标方块靠近底座方块 → 1 - tanh(距离/std)。"""
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    goal = env.scene[det.cube_to_stack_on].data.root_pos_w
    d = torch.linalg.norm(cube[:, :2] - goal[:, :2], dim=1)   # 水平距离
    return 1.0 - torch.tanh(d / std)


def object_stacked_reward(env) -> torch.Tensor:
    """堆叠奖励（分阶段 RL 用）：方块成功堆叠 → 1.0。"""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    return mdp.object_stacked(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        upper_object_cfg=SceneEntityCfg(det.cube_to_grasp),
        lower_object_cfg=SceneEntityCfg(det.cube_to_stack_on),
    ).float()


def object_stacked_dense_reward(env, std_xy: float = 0.08, std_z: float = 0.03,
                                cube_height: float = 0.05) -> torch.Tensor:
    """稠密堆叠奖励（分阶段 RL 用，替代稀疏 object_stacked）。

    奖励 = 水平对齐(1-tanh(d_xy/std)) × 高度对齐(1-tanh(d_z/std))，
    两个都接近才高——给策略连续梯度引导"把方块搬到目标正上方且到堆叠高度"。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    base = env.scene[det.cube_to_stack_on].data.root_pos_w
    d_xy = torch.linalg.norm(cube[:, :2] - base[:, :2], dim=1)
    z_target = base[:, 2] + cube_height               # 方块要落在底座方块顶部
    d_z = torch.abs(cube[:, 2] - z_target)
    return (1.0 - torch.tanh(d_xy / std_xy)) * (1.0 - torch.tanh(d_z / std_z))


def _shaping_potential(env) -> torch.Tensor:
    """非饱和势能 ``φ = -(d_approach + d_stack)``。

    **回归修复（M1 实验发现）**：旧实现用各阶段进度列之和作势能，其中 approach 列
    ``clamp(1 - d/0.15)`` 在目标附近**饱和**——靠近目标时随机游走只能降不能升，
    产生持续的负塑形偏置（200 迭代 mean reward 被拉到 -2.68，agent 学不动）。
    改用距离势能（非饱和、随接近单调上升），随机游走期望为 0，接近才给正信号。

    语义计划驱动：未规划阶段对应的距离项不参与（如 pick-only 不带 d_stack）。
    """
    ee_w, cube_pos, _grasp, _stacked = _stage_signals_from_env(env)
    det = _get_detector()
    held = cube_pos[det.cube_to_grasp]
    base = cube_pos[det.cube_to_stack_on]

    d_app = torch.hypot(ee_w[:, 0] - held[:, 0], ee_w[:, 1] - held[:, 1])
    d_stack = torch.hypot(ee_w[:, 0] - base[:, 0], ee_w[:, 1] - base[:, 1])

    active = _active_stages
    use_app = active is None or "approach" in active
    use_stack = active is None or ("move" in active or "stack" in active)
    return -(d_app if use_app else torch.zeros_like(d_app)) \
           - (d_stack if use_stack else torch.zeros_like(d_stack))


def build_stage_rewards_cfg(
    weights: dict,
    thresholds: dict | None = None,
    stages: list[str] | None = None,
    cube_to_grasp: str = "cube_2",
    cube_to_stack_on: str = "cube_1",
    active_stages: list[str] | None = None,
) -> object:
    """构建带阶段感知奖励的 ``RewardsCfg``（需 Isaac 环境导入 isaaclab）。

    Args:
        weights: config 的 reward_weights（含 action_penalty / progress_shaping / 各阶段名）
        thresholds: config 的 thresholds（阶段切换阈值）
        stages: 阶段名列表（缺省取 detector 默认五阶段）
        cube_to_grasp, cube_to_stack_on: 目标方块 / 底座方块名（**必须来自 config task**，
            否则阶段检测会盯错方块——旧版写死默认值的隐患）
        active_stages: 指令覆盖的活动阶段（语义计划驱动；None = 全部阶段）

    Returns:
        一个 ``isaaclab.managers.RewardTermCfg`` 集合，可作为 ``env_cfg.rewards`` 使用。
    """
    from isaaclab.envs import mdp
    from isaaclab.managers import RewardTermCfg as RewardTerm
    from isaaclab.utils.configclass import configclass

    global _detector, _active_stages
    _detector = StageDetector(
        stages=stages,
        thresholds=thresholds or {},
        cube_to_grasp=cube_to_grasp,
        cube_to_stack_on=cube_to_stack_on,
    )
    _active_stages = list(active_stages) if active_stages is not None else None

    @configclass
    class StageRewardsCfg:
        """阶段感知稠密奖励配置。"""

        action_penalty = RewardTerm(func=mdp.action_l2, weight=weights["action_penalty"])
        stage_progress = RewardTerm(
            func=stage_potential_reward,
            params={"weights": weights, "gamma": 0.99},
            weight=1.0,
        )
        stage_transition = RewardTerm(
            func=stage_transition_reward,
            params={"weights": weights},
            weight=1.0,
        )
        lifting_object = RewardTerm(
            func=object_is_lifted_reward,
            params={"minimal_height": (thresholds or {}).get("lift_min_height", 0.04)},
            weight=weights.get("lift_object", 15.0),
        )

    return StageRewardsCfg()
