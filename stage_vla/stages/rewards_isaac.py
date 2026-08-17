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
from . import grasp as grasp_geo
from .detector import StageDetector
from .grasp import red_on_blue_frame
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


# ============================================================================
# 真实抓取状态（physical_grasp / stable_grasp / drop）—— env.extras 缓存共享
# ============================================================================
def _physical_grasp(env) -> torch.Tensor:
    """物理抓取掩码 ``[N]`` bool：优先接触传感器，缺省用几何代理。

    - 几何代理：方块在两指之间 + 手指未全开 + 末端贴近
      （``grasp.physical_grasp_geometric``，见交接文档 3/4 节：不要用 ``finger<0.01``）；
    - 若 cfg_surgery 注入了左右指 ContactSensor（``scene.lf_contact / rf_contact``），
      再叠加"两指接触力 > 阈值"（过滤体为被夹方块），更接近物理真值。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    robot = env.scene["robot"]
    # 指尖用 FrameTransformer 的 tool_leftfinger/tool_rightfinger 帧（含 0.046 偏移）。
    # 关键：body_pos_w 的 panda_leftfinger/rightfinger 是手指**根/枢轴**（近手心），
    # 用它算"方块在两指之间"几何判定永远不成立（旧 diag 两个 policy 的 physical 恒 0）。
    ee_frames = env.scene["ee_frame"].data.target_pos_w      # [N,3,3]: ee/right/left
    lf = ee_frames[:, 2, :]    # tool_leftfinger 指尖
    rf = ee_frames[:, 1, :]    # tool_rightfinger 指尖
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    ee = ee_frames[:, 0, :]
    finger = robot.data.joint_pos[:, 7:9]

    geom = grasp_geo.physical_grasp_geometric(lf, rf, cube, ee, finger)
    if hasattr(env.scene, "lf_contact") and hasattr(env.scene, "rf_contact"):
        lf_f = env.scene["lf_contact"].data.force_matrix_w[:, 0, 0].norm(dim=-1)
        rf_f = env.scene["rf_contact"].data.force_matrix_w[:, 0, 0].norm(dim=-1)
        return geom & (lf_f > 1.0) & (rf_f > 1.0)
    return geom


# 帧级缓存：同一 env.step 内多次调用只计算一次。reward manager 会**跳过 weight==0 的
# reward term**（isaaclab/managers/reward_manager.py），因此 gated reward 各自调用
# 本函数，必须幂等——否则"连续抓取计数器"会在同一帧被重复累加。
_grasp_cache: dict[int, tuple[int, tuple[torch.Tensor, torch.Tensor]]] = {}
# stable_grasp 判定步数（同时用于 state_bank 的"首次达标"采集标志）
STABLE_GRASP_STEPS = 20


def _grasp_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """计算并缓存抓取状态，返回 ``(physical, counter)``（同帧幂等）。

    - ``physical``：本步物理抓取 ``[N]`` bool；
    - ``counter``：physical 连续保持步数 ``[N]`` long（中断/首帧清零）；
    同时缓存 ``_stage_prev_physical / _stage_drop / _stage_grasp_counter``
    与 ``_stage_grasp_just_stable``（state_bank 采集用，见 :mod:`state_bank`）。
    """
    env = getattr(env, "unwrapped", env)
    step_key = int(env.common_step_counter)          # 每 env.step 全局 +1，作帧键
    cached = _grasp_cache.get(id(env))
    if cached is not None and cached[0] == step_key:
        return cached[1]

    physical = _physical_grasp(env)
    first = first_frame_mask(env.episode_length_buf)

    prev = env.extras.get("_stage_prev_physical")
    prev = torch.zeros_like(physical) if prev is None else prev.to(physical.device)
    prev = torch.where(first, physical, prev)          # 首帧对齐（视为本步已抓住=无掉落）
    drop = prev & (~physical)

    counter = env.extras.get("_stage_grasp_counter")
    counter = torch.zeros_like(physical, dtype=torch.long) if counter is None else counter.to(physical.device)
    counter = torch.where(first, torch.zeros_like(counter), counter)   # 首帧清零
    counter = torch.where(physical, counter + 1, torch.zeros_like(counter))

    env.extras["_stage_prev_physical"] = physical
    env.extras["_stage_drop"] = drop
    env.extras["_stage_grasp_counter"] = counter
    env.extras["_stage_physical_grasp"] = physical
    env.extras["_stage_grasp_just_stable"] = (counter == STABLE_GRASP_STEPS) & (counter > 0)
    _grasp_cache[id(env)] = (step_key, (physical, counter))
    return physical, counter


def _stable_grasp_mask(env, stable_steps: int = STABLE_GRASP_STEPS) -> torch.Tensor:
    """本步 stable_grasp：连续物理抓取 >= stable_steps 步（同帧幂等）。"""
    env = getattr(env, "unwrapped", env)
    _physical, counter = _grasp_state(env)
    return counter >= stable_steps


def drop_penalty_reward(env) -> torch.Tensor:
    """掉块惩罚：上一步物理抓取成立、本步丢失 → -1（权重在 config 里乘）。

    让策略学会"抓住就保持"——lift/move/stack 阶段的连续性是成功前提
    （交接文档 6.1 节：抬高不是目的，抓着抬高才是）。
    """
    env = getattr(env, "unwrapped", env)
    _physical, _counter = _grasp_state(env)
    drop = env.extras.get("_stage_drop", torch.zeros_like(_physical)).to(_physical.device)
    return -drop.float()


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


def object_is_lifted_reward(env, minimal_height: float = 0.04, stable_steps: int = 20) -> torch.Tensor:
    """方块被抬起奖励（**grasp-gated** 版，替代官方 Isaac-Lift 阈值判定）。

    旧版只判 ``cube.z > 阈值``，被撞飞/滚高也能骗分（success=0 的帮凶之一）；
    改为 ``stable_grasp * clamp((cube_z - 桌面)/minimal_height, 0, 1)``——
    **只有抓住并抬高才算 lift 进度**（交接文档 6 节）。``minimal_height``
    语义从"判定阈值"变为"目标抬升高度"。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    stable = _stable_grasp_mask(env, stable_steps)
    progress = torch.clamp((cube[:, 2] - calculator.TABLE_Z) / max(minimal_height, 1e-6), 0.0, 1.0)
    return stable.float() * progress


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


def object_grasp_combined_reward(
    env,
    close_scale: float = 3.0,
    grasp_scale: float = 2.0,
    opp_scale: float = 0.3,
    height_scale: float = 1.5,
) -> torch.Tensor:
    """组合抓取奖励（重写版）= 对侧定位 + 高度对齐 + 部分闭合 + 物理抓取。

    旧版 ``closed = (finger < 0.01)`` 奖励夹爪闭死——真夹住方块时手指闭不到
    0.01，奖励方向反了（空手全闭骗分，交接文档 3/4 节）。新版四项（分层引导）：
      - ``opp``：对侧定位（稠密，防"手指同侧"局部最优，**压权重**避免免费 hover）；
      - ``height_align``：指尖高度对准方块中心（引导**下降**——opp 悬空也≈1，
        不推动下降，这是 500/600 迭代策略停在 hover 的根因）；
      - ``closing``：手指**部分闭合** × 末端贴近（不再要求闭死，允许方块在两指间）；
      - ``physical``：物理抓取（对侧 + 部分闭合 + 贴近，稀疏但诚实的完成信号）。

    **重平衡（diag/短训实测）**：``opp`` 单独拿满 ~1.0 使策略停在"方块对侧 hover"，
    压到 0.3 并把"下降"显式变成 height_align 子目标（hover 0.3 vs 全抓 ~5）。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    robot = env.scene["robot"]
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    ee_frames = env.scene["ee_frame"].data.target_pos_w
    ee = ee_frames[:, 0, :]
    d = torch.linalg.norm(cube - ee, dim=-1)

    finger = robot.data.joint_pos[:, 7:9]

    # 指尖用 FrameTransformer 帧，勿用 finger 根（body_pos_w 的 panda_leftfinger 是根/枢轴）
    lfinger = ee_frames[:, 2, :]    # tool_leftfinger
    rfinger = ee_frames[:, 1, :]    # tool_rightfinger

    # 对侧定位（稠密，压权重避免免费 hover）
    vec_l, vec_r = lfinger - cube, rfinger - cube
    direction = (vec_l * vec_r).sum(-1)
    opp = 1.0 - torch.tanh(direction) * (vec_l.norm(-1) + vec_r.norm(-1))

    # 高度对齐：指尖 z 对准方块中心 z（水平近时才给），引导下降
    d_xy = torch.linalg.norm(cube[:, :2] - lfinger[:, :2], dim=-1)
    horiz_near = 1.0 - torch.tanh(d_xy / 0.1)
    height_align = (1.0 - torch.tanh((lfinger[:, 2] - cube[:, 2]).abs() / 0.02)) * horiz_near

    # 部分闭合（**方块在两指之间** 才给分——3000 迭代实测：不门控 between 时，策略
    # 学会"下降+闭合空手"骗 close 分，value≈0.88 但 physical=0）
    between = grasp_geo.cube_between_fingers(lfinger, rfinger, cube).float()
    closing = grasp_geo.fingers_closing(finger).float()
    close = between * closing * (1.0 - torch.tanh(d / 0.05))

    physical = _physical_grasp(env).float()
    return opp_scale * opp + height_scale * height_align + close_scale * close + grasp_scale * physical


def object_near_goal_reward(env, std: float = 0.1, stable_steps: int = 20) -> torch.Tensor:
    """移动奖励（**grasp-gated** 版）：``stable_grasp * (1 - tanh(d/std))``。

    ``d`` 是**被夹方块 cube_2 → 底座 cube_1** 的水平距离（非末端→底座，
    交接文档 7 节）：只有抓着红块朝蓝块移动才有分，红块掉了空手走不骗分。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    goal = env.scene[det.cube_to_stack_on].data.root_pos_w
    stable = _stable_grasp_mask(env, stable_steps)
    d = torch.linalg.norm(cube[:, :2] - goal[:, :2], dim=-1)   # 水平距离
    return stable.float() * (1.0 - torch.tanh(d / std))


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
                                cube_height: float = 0.05, release_scale: float = 5.0,
                                vel_threshold: float = 0.1, stable_steps: int = 20) -> torch.Tensor:
    """稠密堆叠奖励（**grasp-gated + release** 版，交接文档 9 节）。

    两部分：
    - ``held_align``：**抓着**并对齐（红块到蓝块正上方 + 堆叠高度），稠密引导搬运；
    - ``released_stack``：到位 + 已松开 + 低速 → 真堆叠（稀疏高奖）。
    旧版只算几何对齐，机械手悬停夹持也能拿高分但从未真正 release（success 恒 0）。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    cube = env.scene[det.cube_to_grasp].data.root_pos_w
    base = env.scene[det.cube_to_stack_on].data.root_pos_w
    stable = _stable_grasp_mask(env, stable_steps)
    physical, _ = _grasp_state(env)

    d_xy = torch.linalg.norm(cube[:, :2] - base[:, :2], dim=-1)
    z_target = base[:, 2] + cube_height               # 方块要落在底座方块顶部
    d_z = torch.abs(cube[:, 2] - z_target)
    align = (1.0 - torch.tanh(d_xy / std_xy)) * (1.0 - torch.tanh(d_z / std_z))

    held_align = stable.float() * align
    vel = env.scene[det.cube_to_grasp].data.root_lin_vel_w.norm(dim=-1)
    released = (~physical).float() * align * (vel < vel_threshold).float()
    return held_align + release_scale * released


def _shaping_potential(env) -> torch.Tensor:
    """非饱和势能 ``φ = -(d_approach + d_stack)``。

    **回归修复（M1 实验发现）**：旧实现用各阶段进度列之和作势能，其中 approach 列
    ``clamp(1 - d/0.15)`` 在目标附近**饱和**——靠近目标时随机游走只能降不能升，
    产生持续的负塑形偏置（200 迭代 mean reward 被拉到 -2.68，agent 学不动）。
    改用距离势能（非饱和、随接近单调上升），随机游走期望为 0，接近才给正信号。

    **回归修复（交接文档 7 节）**：``d_stack`` 改用**被夹方块 cube_2 → 底座 cube_1**
    的水平距离，而不是末端 → 底座。旧版末端逼近底座会在"红块掉了、空手朝蓝块走"
    时继续给塑形分（reward hacking）。用方块间距后，未抓住时方块间距恒定、无梯度，
    且塑形是连续量（无 grasp 切换的势能跳变）。

    语义计划驱动：未规划阶段对应的距离项不参与（如 pick-only 不带 d_stack）。
    """
    ee_w, cube_pos, _grasp, _stacked = _stage_signals_from_env(env)
    det = _get_detector()
    held = cube_pos[det.cube_to_grasp]
    base = cube_pos[det.cube_to_stack_on]

    d_app = torch.hypot(ee_w[:, 0] - held[:, 0], ee_w[:, 1] - held[:, 1])
    d_stack = torch.hypot(held[:, 0] - base[:, 0], held[:, 1] - base[:, 1])   # cube_2 → cube_1

    active = _active_stages
    use_app = active is None or "approach" in active
    use_stack = active is None or ("move" in active or "stack" in active)
    return -(d_app if use_app else torch.zeros_like(d_app)) \
           - (d_stack if use_stack else torch.zeros_like(d_stack))


def red_on_blue_success(
    env,
    stable_steps: int = STABLE_GRASP_STEPS,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.006,
    cube_height: float = 0.05,
    vel_threshold: float = 0.1,
    gripper_atol: float = 1e-3,
) -> torch.Tensor:
    """**项目自定 success 终止**：红块在蓝块上 + 夹爪释放 + 低速 + 持续 stable_steps。

    替代官方 ``cubes_stacked``（三块塔，与项目 red-on-blue 目标叠放方向相反，
    导致 success 恒 0，见交接文档 2 节）。返回 ``[N]`` bool；连续 ``stable_steps``
    帧满足 :func:`grasp.red_on_blue_frame` 即终止。

    .. note::
        参数必须全部显式（带默认值），不能用 ``**kwargs``——Isaac Lab 的
        ``_resolve_common_term_cfg`` 会把 ``**kwargs`` 解析成**必填参数**导致报错。
    """
    env = getattr(env, "unwrapped", env)
    det = _get_detector()
    upper = env.scene[det.cube_to_grasp].data.root_pos_w
    lower = env.scene[det.cube_to_stack_on].data.root_pos_w
    robot = env.scene["robot"]
    finger = robot.data.joint_pos[:, 7:9]
    vel = env.scene[det.cube_to_grasp].data.root_lin_vel_w

    frame = red_on_blue_frame(
        upper, lower, finger, vel,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        cube_height=cube_height,
        vel_threshold=vel_threshold,
        gripper_atol=gripper_atol,
    )
    first = first_frame_mask(env.episode_length_buf)
    counter = env.extras.get("_stage_stack_counter")
    counter = torch.zeros_like(frame, dtype=torch.long) if counter is None else counter.to(frame.device)
    counter = torch.where(first, torch.zeros_like(counter), counter)   # 首帧清零
    counter = torch.where(frame, counter + 1, torch.zeros_like(counter))
    env.extras["_stage_stack_counter"] = counter
    return counter >= stable_steps


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
        """阶段感知稠密奖励配置。

        .. note::
            gated reward（lifting_object / drop_penalty）依赖本步的 physical/stable
            grasp，由 ``_grasp_state`` **同帧幂等**计算（基于 ``common_step_counter``
            缓存），不需要额外占位 term——reward manager 会跳过 weight==0 的 term。
        """

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
        drop_penalty = RewardTerm(
            func=drop_penalty_reward,
            weight=weights.get("drop_penalty", 10.0),
        )

    return StageRewardsCfg()
