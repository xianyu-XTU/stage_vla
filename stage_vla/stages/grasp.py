"""grasp.py —— 真实抓取判定（纯几何层，可无 Isaac 单测）。

交接结论（见 `E:/STAGE_VLA_SUCCESS_ZERO_FIX_HANDOFF.md`）：
- 官方 ``object_grasped`` 只查"方块距末端 <6cm 且手指未全开"，是几何近似**不是**物理抓取；
- 项目旧奖励 ``finger < 0.01`` 奖励"夹爪闭死"，但真夹住方块时手指闭不到 0.01，
  奖励方向反而反了（没抓到的空手能全闭，抓到反而拿不到分）。

本模块定义"真实抓取"的**几何代理**（纯张量，无仿真依赖）：
- :func:`cube_between_fingers` —— 方块中心投影到"左指→右指"缝隙轴之间、且离轴
  横向偏差小（手指在方块两侧，对侧夹持，排除同侧局部最优）；
- :func:`fingers_closing` —— 手指已部分闭合（未全开），排除"张开空手"；
- :func:`physical_grasp_geometric` —— 上述两者 + 末端贴近方块。

Isaac 层的接触传感器路径（ContactSensorCfg，更接近物理真值）在
:mod:`stage_vla.stages.rewards_isaac` 的 :func:`_physical_grasp` 中接入，本模块
只负责纯几何，保证可单测。
"""

from __future__ import annotations

import torch

# Franka 并行二指夹爪（IK-Rel 任务）的开关值：全开 / 全闭
GRIPPER_OPEN_VAL = 0.04
GRIPPER_CLOSE_VAL = 0.0


def cube_between_fingers(
    lf_pos: torch.Tensor,
    rf_pos: torch.Tensor,
    cube_pos: torch.Tensor,
    perp_max: float = 0.03,
) -> torch.Tensor:
    """方块中心是否位于两指之间的缝隙里（几何判定）。

    把 ``lf→rf`` 连线当作"手指缝隙轴"：
    - 方块中心在该轴上的投影严格位于左指与右指投影**之间**（两指分居方块两侧）；
    - 方块中心到该轴的横向距离 < ``perp_max``（排除方块架在指尖上方/下方）。

    Args:
        lf_pos / rf_pos: 左右指尖位置 ``[N,3]``（world）
        cube_pos: 方块中心 ``[N,3]``
        perp_max: 方块中心到缝隙轴的允许横向偏差（米）

    Returns:
        ``[N]`` bool —— 两指在方块两侧
    """
    gap = rf_pos - lf_pos
    gap_len = gap.norm(dim=-1).clamp(min=1e-6)
    u = gap / gap_len.unsqueeze(-1)                        # 缝隙轴单位向量

    rel = cube_pos - lf_pos
    along = (rel * u).sum(dim=-1)                          # 方块沿轴投影（左指=0，右指=gap_len）
    perp = (rel - along.unsqueeze(-1) * u).norm(dim=-1)    # 方块到缝隙轴的横向距离

    return (along > 0.0) & (along < gap_len) & (perp < perp_max)


def fingers_closing(
    finger_joint_pos: torch.Tensor,
    open_val: float = GRIPPER_OPEN_VAL,
    close_margin: float = 0.005,
) -> torch.Tensor:
    """手指是否已部分闭合（未全开）。输入 ``[N,2]``，输出 ``[N]`` bool。

    "夹住方块"时手指不会闭到 0.01，只要两指都**未处于全开**即视为在闭合/持物。
    """
    return (finger_joint_pos < open_val - close_margin).all(dim=-1)


def physical_grasp_geometric(
    lf_pos: torch.Tensor,
    rf_pos: torch.Tensor,
    cube_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    finger_joint_pos: torch.Tensor,
    *,
    open_val: float = GRIPPER_OPEN_VAL,
    close_margin: float = 0.005,
    ee_max: float = 0.06,
    perp_max: float = 0.03,
) -> torch.Tensor:
    """几何代理"物理抓取"掩码 ``[N]`` bool。

    条件（全部满足才算物理抓取）：
    1. 方块在两指之间（:func:`cube_between_fingers`，对侧夹持）；
    2. 手指已部分闭合（非全开，排除张开空手）；
    3. 末端贴近方块（``< ee_max``，兜底位置合理性）。

    .. note::
        这是**几何代理**；若 cfg_surgery 注入了左右指 ContactSensor，
        ``rewards_isaac._physical_grasp`` 会在此基础上再叠加接触力条件。
    """
    between = cube_between_fingers(lf_pos, rf_pos, cube_pos, perp_max=perp_max)
    closing = fingers_closing(finger_joint_pos, open_val=open_val, close_margin=close_margin)
    near = torch.linalg.norm(cube_pos - ee_pos, dim=-1) < ee_max
    return between & closing & near


def red_on_blue_geometry(
    upper_pos: torch.Tensor,
    lower_pos: torch.Tensor,
    gripper_joint_pos: torch.Tensor,
    *,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.006,
    cube_height: float = 0.05,
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_atol: float = 1e-3,
) -> torch.Tensor:
    """项目自定"红块在蓝块上"几何：堆叠对齐 + 夹爪已释放。

    与官方 ``object_stacked`` 同款几何（上块高于下块约 ``cube_height`` 且 xy 对齐），
    但配对为 ``cube_2 → cube_1``（项目目标 red-on-blue），且要求夹爪**已打开**
    （真释放，不是悬停夹持）。官方 ``cubes_stacked`` 是三块塔（蓝压红压绿）且
    **与项目 red-on-blue 目标叠放方向相反**，导致 success 恒 0 —— 本函数是替代。

    Args:
        upper_pos: 上块（红 cube_2）位置 ``[N,3]``
        lower_pos: 下块（蓝 cube_1）位置 ``[N,3]``
        gripper_joint_pos: 两指关节位置 ``[N,2]``
    Returns:
        ``[N]`` bool —— 单帧"对齐 + 已释放"
    """
    pos_diff = upper_pos - lower_pos
    xy_dist = torch.linalg.norm(pos_diff[..., :2], dim=-1)
    h_dist = torch.linalg.norm(pos_diff[..., 2:], dim=-1)

    stacked = (xy_dist < xy_threshold) & ((h_dist - cube_height) < height_threshold)

    # 夹爪已打开（release）——兼容 [N,2] 与 [2]（单环境切片）
    open_val = torch.full_like(gripper_joint_pos, gripper_open_val)
    released = torch.isclose(gripper_joint_pos[..., 0], open_val[..., 0], atol=gripper_atol) \
        & torch.isclose(gripper_joint_pos[..., 1], open_val[..., 1], atol=gripper_atol)

    return stacked & released


def red_on_blue_frame(
    upper_pos: torch.Tensor,
    lower_pos: torch.Tensor,
    gripper_joint_pos: torch.Tensor,
    upper_lin_vel: torch.Tensor,
    *,
    vel_threshold: float = 0.1,
    **geom_kwargs,
) -> torch.Tensor:
    """单帧"红块在蓝块上且稳定"：几何堆叠 + 释放 + 方块低速。

    低速（``|upper_lin_vel| < vel_threshold``）排除刚掉落后还在晃动/滑落的假成功。
    持续多帧的判定由 Isaac 层（``rewards_isaac.red_on_blue_success``）用计数器实现。
    """
    geom = red_on_blue_geometry(upper_pos, lower_pos, gripper_joint_pos, **geom_kwargs)
    slow = upper_lin_vel.norm(dim=-1) < vel_threshold
    return geom & slow
