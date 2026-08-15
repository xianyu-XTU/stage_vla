"""unsupervised.py —— 无监督阶段分离器（模块① 研究项，M1 实现）。

申请书承诺：基于末端执行器几何约束，**无需人工标注**地将长程任务拆成语义子阶段。

当前 M0 使用 :class:`~stage_vla.stages.detector.StageDetector` 的几何启发式
（阈值由 ``config/thresholds`` 手工给定）。本模块是它的**学习式升级接口**：

- 在轨迹特征空间（末端速度拐点、接触状态、抬升高度）上聚类出阶段中心；
- 拟合出的中心可用于替换或校准几何阈值，从而脱离人工标注。

M1 验收：在带标注的合成轨迹上，聚类中心与几何 detect 的一致性 > 阈值，
随后由 :meth:`fit_unsupervised_centers` 产出可写入 config 的阈值。
"""

from __future__ import annotations

import numpy as np
import torch


def fit_unsupervised_centers(
    trajectories: torch.Tensor | np.ndarray,
    n_stages: int | None = None,
    *,
    feature_cols: list[int] | None = None,
) -> torch.Tensor:
    """根据轨迹特征学习阶段中心 —— 研究扩展占位。

    Args:
        trajectories: ``[T, N, D]`` 或 ``[T, D]`` 轨迹特征（见 feature_cols）
        n_stages: 期望的阶段数（默认取 config 的 stages 数）
        feature_cols: 参与聚类的特征列（末端速度 / 接触 / 抬升高度）

    Returns:
        ``[n_stages, D]`` 聚类中心（尚未实现，M1 落地）
    """
    raise NotImplementedError(
        "无监督阶段分离器为 M1 研究项，尚未实现。当前使用几何启发式 "
        "StageDetector（阈值见 config/thresholds）。参见 docs/module1_stages.md。"
    )
