"""detector.py —— 阶段检测器（几何启发式，全向量化）。

将长程堆叠任务划分为 ``[approach, grasp, lift, move, stack]`` 五阶段。
判定逻辑收敛在 :mod:`stage_vla.stages.calculator` 的纯函数里，本类只做两件事：

1. 提供可复用的几何判定原语（``is_near / is_grasped / is_stacked / is_lifted / is_aligned``）；
2. 包装 calculator，用构造时传入的 ``stages / thresholds / 目标方块名`` 驱动
   :meth:`detect` 与 :meth:`progress`。

回归点（吸收旧工程教训）：
- ``is_stacked`` **独立于** ``is_grasped``（旧版误用 ``is_grasped(stacked_signal)`` 判堆叠）。
- 阶段判定为**逆序布尔链**（stack → move → lift → grasp → approach）。
"""

from __future__ import annotations

import torch

from . import calculator

# 默认阶段序列（与 config/default.yaml 保持一致；可经 from_settings 覆盖）
DEFAULT_STAGES = ("approach", "grasp", "lift", "move", "stack")

# 默认阈值（与 config/default.yaml 的 thresholds 一致；缺省时兜底，避免空 dict KeyError）
DEFAULT_THRESHOLDS = {
    "approach_dist": 0.15,
    "grasp_reward_thresh": 0.5,
    "lift_height": 0.06,
    "place_align_dist": 0.05,
}


class StageDetector:
    """几何阶段检测器。

    Parameters
    ----------
    stages : 阶段名序列（默认五阶段）
    thresholds : 含 approach_dist / grasp_reward_thresh / lift_height / place_align_dist
    cube_to_grasp, cube_to_stack_on : 目标方块 / 底座方块名
    """

    def __init__(
        self,
        stages: list[str] | None = None,
        thresholds: dict | None = None,
        cube_to_grasp: str = "cube_2",
        cube_to_stack_on: str = "cube_1",
    ):
        self.stages = list(stages or DEFAULT_STAGES)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.cube_to_grasp = cube_to_grasp
        self.cube_to_stack_on = cube_to_stack_on

    @classmethod
    def from_settings(cls, settings) -> "StageDetector":
        """从 :class:`~stage_vla.core.config.Settings` 构造（推荐入口）。"""
        return cls(
            stages=settings.stages,
            thresholds=settings.thresholds,
            cube_to_grasp=settings.task["cube_to_grasp"],
            cube_to_stack_on=settings.task["cube_to_stack_on"],
        )

    # ------------------------------------------------------------------
    # 几何判定原语（输入输出 torch 张量，[N] 或 [N,3]）
    # ------------------------------------------------------------------
    @staticmethod
    def is_near(ee_pos: torch.Tensor, target_pos: torch.Tensor, dist: float) -> torch.Tensor:
        """末端执行器是否靠近目标（水平距离 < dist）。"""
        return calculator._is_near(ee_pos, target_pos, dist)

    @staticmethod
    def is_grasped(grasp_signal: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """目标方块是否被抓取（信号 > 阈值）。"""
        return grasp_signal > thresh

    @staticmethod
    def is_stacked(stacked_signal: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """堆叠是否完成（信号 > 阈值）。【独立于 is_grasped —— 回归点】"""
        return stacked_signal > thresh

    def is_lifted(self, held_pos: torch.Tensor, height: float | None = None) -> torch.Tensor:
        """方块是否被抬起超过阈值高度。"""
        h = self.thresholds.get("lift_height", 0.06) if height is None else height
        return held_pos[..., 2] > calculator.TABLE_Z + h

    def is_aligned(self, held_pos: torch.Tensor, base_pos: torch.Tensor, dist: float | None = None) -> torch.Tensor:
        """被夹方块与底座方块是否水平对齐（'move' 阶段）。"""
        d = self.thresholds.get("place_align_dist", 0.05) if dist is None else dist
        dx = held_pos[..., 0] - base_pos[..., 0]
        dy = held_pos[..., 1] - base_pos[..., 1]
        return (dx**2 + dy**2) < d**2

    # ------------------------------------------------------------------
    # 阶段判定（委托 calculator，单一实现）
    # ------------------------------------------------------------------
    def detect(
        self,
        ee_pos: torch.Tensor,
        cube_positions: dict[str, torch.Tensor],
        grasp_signal: torch.Tensor,
        stacked_signal: torch.Tensor,
    ) -> torch.Tensor:
        """返回当前阶段索引 ``[N]`` LongTensor。"""
        return calculator.signals_to_stage(
            ee_pos, cube_positions, grasp_signal, stacked_signal,
            stages=self.stages, thresholds=self.thresholds,
            cube_to_grasp=self.cube_to_grasp, cube_to_stack_on=self.cube_to_stack_on,
        )

    def progress(
        self,
        ee_pos: torch.Tensor,
        cube_positions: dict[str, torch.Tensor],
        grasp_signal: torch.Tensor,
        stacked_signal: torch.Tensor,
    ) -> torch.Tensor:
        """返回各阶段完成度 ``[N, n_stages]``（0~1），用于势能塑形。"""
        return calculator.signals_to_progress(
            ee_pos, cube_positions, grasp_signal, stacked_signal,
            stages=self.stages, thresholds=self.thresholds,
            cube_to_grasp=self.cube_to_grasp, cube_to_stack_on=self.cube_to_stack_on,
        )

    def stage_names(self, stage_indices: torch.Tensor) -> list[str]:
        """把阶段索引张量转成阶段名字符串列表（调试用）。"""
        return [self.stages[int(i)] for i in stage_indices.tolist()]
