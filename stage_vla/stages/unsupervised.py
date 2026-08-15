"""unsupervised.py —— 无监督阶段分离器（模块① 核心增量，M1 实现）。

申请书承诺：基于机器人末端执行器几何约束，**无需人工标注**地将长程任务拆成语义子阶段。
本项目对标 STARE-VLA 的"任务专属规则式分离器"（每任务手工设计事件阈值），用数据驱动
的无监督聚类自动发现阶段边界，是区别于最相关工作的核心竞争力。

方法（吸收 HiTS / STARE-VLA 教训）：
1. **特征工程**：逐时间步提取末端执行器几何约束特征
   - 末端速度幅值、末端↔目标方块水平距离、被夹方块抬升高度、抓取信号、末端↔底座距离；
2. **无监督分段**：对全部轨迹时间步做 KMeans 聚类（K=阶段数），
   再按"任务进度代理"（抬升高度 + 抓取 + 距离补数的加权和）对簇**单调排序**，
   得到与任务顺序一致的无监督阶段中心；
3. **分配/评估**：最近簇分配阶段；与几何判定（StageDetector）对比一致性。

只依赖 torch（Isaac Sim kit python 自带），不引入 sklearn。
"""

from __future__ import annotations

import torch

from ..core.errors import StageVLAError
from . import calculator
from .detector import DEFAULT_STAGES

# 特征列索引（D=5）
F_EE_VEL, F_GRASP_DIST, F_LIFT_HEIGHT, F_GRASP, F_STACK_DIST = range(5)
FEATURE_NAMES = ("ee_vel", "grasp_dist", "lift_height", "grasp", "stack_dist")

# 特征构造参考量（米 / 米每秒；仅用于把原始量纲压到可比较的量级，非阶段阈值）
_REF_VEL = 0.5        # 参考末端速度 m/s
_REF_TABLE_Z = calculator.TABLE_Z


def extract_features(
    ee_pos: torch.Tensor,
    cube_positions: dict[str, torch.Tensor],
    grasp_signal: torch.Tensor,
    *,
    stages: list[str] | None = None,
    thresholds: dict | None = None,
    cube_to_grasp: str = "cube_2",
    cube_to_stack_on: str = "cube_1",
    ee_vel: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐时间步提取末端执行器几何约束特征，返回 ``[T, 5]``。

    Args:
        ee_pos: [T,3] 末端位置
        cube_positions: {name: [T,3]} 各方块位置
        grasp_signal: [T] 抓取信号
        ee_vel: [T] 末端速度幅值（None 时由 ee_pos 一阶差分估算）
    """
    held = cube_positions[cube_to_grasp]
    base = cube_positions[cube_to_stack_on]

    if ee_vel is None:
        vel = torch.zeros_like(ee_pos[:, 0])
        if ee_pos.shape[0] > 1:
            diff = (ee_pos[1:] - ee_pos[:-1]).norm(dim=1)
            vel[1:] = diff
            vel[0] = diff[0] if diff.numel() else 0.0
        ee_vel = vel
    ee_vel = ee_vel.to(ee_pos.dtype)

    d_grasp = torch.hypot(ee_pos[:, 0] - held[:, 0], ee_pos[:, 1] - held[:, 1])
    d_stack = torch.hypot(ee_pos[:, 0] - base[:, 0], ee_pos[:, 1] - base[:, 1])

    features = torch.stack(
        [
            ee_vel / _REF_VEL,                                        # 0 末端速度
            d_grasp,                                                  # 1 距目标方块距离
            (held[:, 2] - _REF_TABLE_Z),                              # 2 抬升高度
            grasp_signal.to(ee_pos.dtype),                            # 3 抓取信号
            d_stack,                                                  # 4 距底座距离
        ],
        dim=1,
    )
    return features


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐列 min-max 归一化到 [0,1]，返回 (归一化, min, span)。"""
    xmin = x.min(dim=0).values
    span = x.max(dim=0).values - xmin
    span = torch.where(span < eps, torch.ones_like(span), span)
    return (x - xmin) / span, xmin, span


def _progress_proxy(features: torch.Tensor) -> torch.Tensor:
    """任务进度代理（越高越接近完成），用于对聚类簇单调排序。

    **只取单调信号**（避免非单调项导致排序错误）：
    - 抓取信号（0→1 后保持）
    - 抬升高度（0→高后保持）
    - ``1 - 末端↔底座距离/参考``（move 阶段单调增大）

    注意：**不能**用"末端↔目标方块距离"——move/stack 阶段末端离开目标方块，该项会回落。
    仅用于排序，不参与聚类本身。
    """
    _ee_vel, _d_grasp, lift, grasp, d_stack = features.unbind(dim=1)
    proxy = (
        grasp.clamp(min=0.0, max=1.0)
        + (lift.clamp(min=0.0, max=0.06) / 0.06)
        + (1.0 - (d_stack / 0.5).clamp(min=0.0, max=1.0))
    )
    return proxy


def _kmeans(features: torch.Tensor, k: int, max_iter: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """标准 Lloyd KMeans（torch 实现）。返回 (centers[K,D], assignments[N])。"""
    if k < 1 or k > features.shape[0]:
        raise StageVLAError(f"k={k} 超出样本数 {features.shape[0]}")
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(features.shape[0], generator=gen)[:k]
    centers = features[idx].clone()

    for _ in range(max_iter):
        dist = torch.cdist(features, centers)
        assign = dist.argmin(dim=1)
        new_centers = torch.stack(
            [features[assign == j].mean(dim=0) if (assign == j).any() else centers[j]
             for j in range(k)]
        )
        if torch.allclose(new_centers, centers, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers
    dist = torch.cdist(features, centers)
    assign = dist.argmin(dim=1)
    return centers, assign


def fit_unsupervised_centers(
    trajectories: torch.Tensor | list[torch.Tensor],
    n_stages: int | None = None,
    feature_cols: list[int] | None = None,
    *,
    max_iter: int = 100,
    seed: int = 0,
) -> dict:
    """从轨迹特征无监督拟合阶段中心（M1 实现）。

    方法：**多维过分割 + 进度排序 + 贪心合并**。
    直接对多维特征做 KMeans(K) 会把"approach 这类在特征空间连续扫过的阶段"拆成多个簇
    （高方差连续轨迹），与阶段数对不上。因此：
    1. 过分割：KMeans(K×2) 先把高方差连续段拆细（approach → 多个簇）；
    2. 排序：按各簇的**任务进度代理**（单调：抓取 + 抬升 + 接近底座）升序排列；
    3. 合并：贪心合并相邻（进度差最小）的簇，直到剩 K 个 → 与任务阶段一一对应。
    全程无任何阶段标注，纯几何信号驱动。阶段中心 = 各段在聚类特征空间的均值。

    Args:
        trajectories: ``[T, N, D]`` 批量，或 ``[N, D]``，或 ``[T, D]`` 列表；D 为特征维
        n_stages: 期望阶段数（缺省取 detector 默认五阶段）
        feature_cols: 参与聚类的特征列子集（None = 全部）
        max_iter / seed: KMeans 超参

    Returns:
        dict:
          centers: [K, D] 无监督阶段中心（按任务进度单调排序）
          assignments: [N_total] 各时间步的阶段索引（映射到排序后顺序）
          proxy_order: 合并前各过分割簇的进度排序（解释用）
          norm: (min, span) 归一化参数（供分配时复用）
    """
    flat_full, _ = _flatten(trajectories)   # 完整特征（含全部 5 列，用于进度代理）
    flat = flat_full if feature_cols is None else flat_full[:, feature_cols]

    k = n_stages or len(DEFAULT_STAGES)
    if flat.shape[0] < k:
        raise StageVLAError(f"轨迹样本数 {flat.shape[0]} 少于阶段数 {k}")

    x, xmin, span = _minmax_normalize(flat)
    proxy = _progress_proxy(flat_full)

    # 1) 过分割
    k_ov = min(max(2 * k, 6), flat.shape[0])
    centers_ov, assign_ov = _kmeans(x, k_ov, max_iter=max_iter, seed=seed)
    c_proxy = torch.stack(
        [proxy[assign_ov == j].mean() if (assign_ov == j).any() else torch.tensor(-1e9)
         for j in range(k_ov)]
    )
    order = torch.argsort(c_proxy)  # 簇按进度升序

    # 2) 贪心合并相邻簇到 k 组（合并进度差最小的相邻对）
    groups = [[c] for c in order.tolist()]
    while len(groups) > k:
        gaps = [
            abs(c_proxy[groups[i][-1]] - c_proxy[groups[i + 1][0]])
            for i in range(len(groups) - 1)
        ]
        idx = gaps.index(min(gaps))
        groups[idx].extend(groups[idx + 1])
        del groups[idx + 1]

    # 3) 重映射：过分割簇 → 阶段（合并后的位置）
    stage_of_cluster = torch.empty(k_ov, dtype=torch.long)
    for stage, group in enumerate(groups):
        for c in group:
            stage_of_cluster[c] = stage
    assignments = stage_of_cluster[assign_ov]

    # 阶段中心 = 各段在聚类特征空间的均值
    centers = torch.stack(
        [x[assignments == stage].mean(dim=0) if (assignments == stage).any() else x.mean(dim=0)
         for stage in range(k)]
    )

    return {
        "centers": centers,
        "assignments": assignments,
        "proxy_order": order,
        "norm": (xmin, span),
    }


def assign_stages(
    features: torch.Tensor,
    centers: torch.Tensor,
    norm: tuple[torch.Tensor, torch.Tensor] | None = None,
    feature_cols: list[int] | None = None,
) -> torch.Tensor:
    """把特征 [T,D] 分配到最近阶段中心，返回 [T] 阶段索引。"""
    x = features
    if feature_cols is not None:
        x = x[:, feature_cols]
    if norm is not None:
        xmin, span = norm
        x = (x - xmin) / span
    dist = torch.cdist(x, centers)
    return dist.argmin(dim=1)


def evaluate_consistency(pred_stage: torch.Tensor, true_stage: torch.Tensor) -> float:
    """无监督阶段划分与真值阶段的一致性（匹配率 ∈ [0,1]）。"""
    pred = pred_stage.flatten()
    true = true_stage.flatten()
    if pred.numel() != true.numel():
        raise StageVLAError("pred/true 长度不一致")
    return float((pred == true).float().mean().item())


def _flatten(trajectories: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, int]:
    """把各种轨迹表示展平成 [N_total, D]。返回 (flat, D)。"""
    if isinstance(trajectories, torch.Tensor):
        t = trajectories
        if t.dim() == 3:      # [T, N, D]
            t = t.permute(1, 0, 2).reshape(-1, t.shape[2])
        elif t.dim() == 1:    # [T]（单特征）
            t = t.unsqueeze(1)
        return t, t.shape[1]
    if isinstance(trajectories, (list, tuple)):
        pieces = [tr.unsqueeze(1) if tr.dim() == 1 else tr for tr in trajectories]
        t = torch.cat([p.reshape(-1, p.shape[-1]) for p in pieces], dim=0)
        return t, t.shape[1]
    raise StageVLAError(f"不支持的轨迹类型：{type(trajectories)}")
