"""calculator.py —— 阶段计算器（纯函数层，单一实现）。

把"几何/接触信号 → 阶段索引 / 阶段完成度"的映射收敛到这一个模块：

- :func:`signals_to_stage`   信号 → 当前阶段索引 ``[N]``
- :func:`signals_to_progress` 信号 → 各阶段完成度 ``[N, K]``

供 :class:`stage_vla.stages.detector.StageDetector`（几何检测）、``rewards``（势能塑形 /
阶段完成奖）、``envs``（Isaac 环境接口）共同复用，**防止同一套阶段判定逻辑在多个
模块各写一份导致漂移**（旧工程核心阶段逻辑只存在于 stage_detector.py 一处，但
rewards 与 semantic 各自隐式假设了阶段顺序，属于隐性漂移源）。

所有函数是纯张量运算，不 import 任何仿真依赖，可无 Isaac Sim 单测。
"""

from __future__ import annotations

import torch

# 桌面高度常量（方块坐标系中桌面 z ≈ 0.02，与旧工程一致）
TABLE_Z = 0.02


def _idx(stages: list[str], name: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(stages.index(name), device=device, dtype=torch.long)


def _is_near(ee_pos: torch.Tensor, target_pos: torch.Tensor, dist: float) -> torch.Tensor:
    dx = target_pos[..., 0] - ee_pos[..., 0]
    dy = target_pos[..., 1] - ee_pos[..., 1]
    return (dx**2 + dy**2) < dist**2


def _is_grasped(signal: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return signal > thresh


def _is_stacked(signal: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return signal > thresh


def _is_lifted(held_pos: torch.Tensor, lift_height: float) -> torch.Tensor:
    return held_pos[..., 2] > TABLE_Z + lift_height


def _is_aligned(held_pos: torch.Tensor, base_pos: torch.Tensor, dist: float) -> torch.Tensor:
    dx = held_pos[..., 0] - base_pos[..., 0]
    dy = held_pos[..., 1] - base_pos[..., 1]
    return (dx**2 + dy**2) < dist**2


def signals_to_stage(
    ee_pos: torch.Tensor,
    cube_positions: dict[str, torch.Tensor],
    grasp_signal: torch.Tensor,
    stacked_signal: torch.Tensor,
    *,
    stages: list[str],
    thresholds: dict,
    cube_to_grasp: str,
    cube_to_stack_on: str,
) -> torch.Tensor:
    """信号 → 当前阶段索引 ``[N]`` LongTensor。

    逆序布尔链判定（stack → move → lift → grasp → approach），优先级从高到低：
    堆叠完成 > 抬起并对齐（移动）> 抓取并抬起 > 已抓取 > 接近。

    Args:
        ee_pos: 末端执行器位置 [N,3]（世界系）
        cube_positions: {"cube_1": [N,3], ...} 各方块位置（世界系）
        grasp_signal: 目标方块被抓取信号 [N]（数值，>0.5 视为已抓取）
        stacked_signal: 堆叠完成信号 [N]（数值，>0.5 视为完成）
        stages: 阶段名列表，如 ["approach","grasp","lift","move","stack"]
        thresholds: 含 approach_dist / grasp_reward_thresh / lift_height / place_align_dist
        cube_to_grasp: 目标方块名
        cube_to_stack_on: 底座方块名
    """
    base = cube_positions[cube_to_stack_on]
    held = cube_positions[cube_to_grasp]
    device = ee_pos.device

    mask_stack = _is_stacked(stacked_signal, thresholds["grasp_reward_thresh"])
    mask_move = _is_lifted(held, thresholds["lift_height"]) & _is_aligned(held, base, thresholds["place_align_dist"])
    mask_lift = _is_grasped(grasp_signal, thresholds["grasp_reward_thresh"]) & _is_lifted(held, thresholds["lift_height"])
    mask_grasp = _is_grasped(grasp_signal, thresholds["grasp_reward_thresh"])

    i_stack = _idx(stages, "stack", device)
    i_move = _idx(stages, "move", device)
    i_lift = _idx(stages, "lift", device)
    i_grasp = _idx(stages, "grasp", device)
    i_approach = _idx(stages, "approach", device)

    return torch.where(
        mask_stack, i_stack,
        torch.where(
            mask_move, i_move,
            torch.where(
                mask_lift, i_lift,
                torch.where(mask_grasp, i_grasp, i_approach),
            ),
        ),
    )


def signals_to_progress(
    ee_pos: torch.Tensor,
    cube_positions: dict[str, torch.Tensor],
    grasp_signal: torch.Tensor,
    stacked_signal: torch.Tensor,
    *,
    stages: list[str],
    thresholds: dict,
    cube_to_grasp: str,
    cube_to_stack_on: str,
) -> torch.Tensor:
    """信号 → 各阶段完成度 ``[N, K]``（0~1），用于势能塑形。

    已完成阶段置 1，当前阶段保留内部进度，未到阶段置 0。
    内部复用 :func:`signals_to_stage` 的结果做掩码，保证阶段判定单一实现。
    """
    base = cube_positions[cube_to_stack_on]
    held = cube_positions[cube_to_grasp]
    device = ee_pos.device
    N = ee_pos.shape[0]
    K = len(stages)

    cur = torch.zeros(N, K, device=device, dtype=torch.float)
    i = {name: stages.index(name) for name in stages}

    # approach 进度：末端到目标方块水平距离的补数
    d_approach = torch.hypot(ee_pos[:, 0] - held[:, 0], ee_pos[:, 1] - held[:, 1])
    cur[:, i["approach"]] = torch.clamp(1.0 - d_approach / thresholds["approach_dist"], min=0.0, max=1.0)
    # grasp
    cur[:, i["grasp"]] = grasp_signal.clamp(min=0.0, max=1.0)
    # lift：方块抬升高度
    cur[:, i["lift"]] = torch.clamp((held[:, 2] - TABLE_Z) / thresholds["lift_height"], min=0.0, max=1.0)
    # move：与底座水平对齐
    d_move = torch.hypot(held[:, 0] - base[:, 0], held[:, 1] - base[:, 1])
    cur[:, i["move"]] = torch.clamp(1.0 - d_move / thresholds["place_align_dist"], min=0.0, max=1.0)
    # stack
    cur[:, i["stack"]] = stacked_signal.clamp(min=0.0, max=1.0)

    # 用 stage 判定做掩码：已过阶段置 1，当前保留，未到置 0
    stage = signals_to_stage(
        ee_pos, cube_positions, grasp_signal, stacked_signal,
        stages=stages, thresholds=thresholds,
        cube_to_grasp=cube_to_grasp, cube_to_stack_on=cube_to_stack_on,
    )
    for k in range(K):
        passed = stage > k
        current = stage == k
        cur[:, k] = torch.where(
            passed,
            torch.ones_like(cur[:, k]),
            torch.where(current, cur[:, k], torch.zeros_like(cur[:, k])),
        )
    return cur
