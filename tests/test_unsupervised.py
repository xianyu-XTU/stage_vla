"""无监督阶段分离器测试：合成带标注的 5 阶段轨迹，验证聚类恢复一致性。

M1 验收标准：聚类恢复的阶段与几何判定一致性 > 阈值（此处 0.7，合成数据留余量）。
"""

from __future__ import annotations

import torch

from stage_vla.stages import unsupervised
from stage_vla.stages.unsupervised import (
    F_GRASP,
    F_GRASP_DIST,
    F_LIFT_HEIGHT,
    F_STACK_DIST,
    evaluate_consistency,
    extract_features,
    fit_unsupervised_centers,
)

TABLE_Z = 0.02
GRASP_CUBE = torch.tensor([0.0, 0.5, TABLE_Z])   # 目标方块（红）
STACK_CUBE = torch.tensor([0.5, 0.0, TABLE_Z])   # 底座（蓝）


def _make_trajectory(t0: int, n_steps: int = 200, seed: int = 0) -> tuple[dict, torch.Tensor]:
    """生成一段 pick-and-place 轨迹，返回 (cube_positions, 真值阶段标签 [T])。

    阶段时间片：approach[0:50) grasp[50:80) lift[80:120) move[120:160) stack[160:200)。
    """
    torch.manual_seed(seed)
    T = n_steps
    # 定义关键点（世界坐标，让 ee 在桌面上方运动）
    home = torch.tensor([-0.3, 0.5, 0.3])
    above_grasp = torch.tensor([0.0, 0.5, 0.25])
    lift_top = torch.tensor([0.0, 0.5, 0.30])
    above_stack = torch.tensor([0.5, 0.0, 0.30])

    def lerp(a, b, alpha):
        return a + (b - a) * alpha

    ee = torch.zeros(T, 3)
    true_stage = torch.zeros(T, dtype=torch.long)
    grasp = torch.zeros(T)

    # approach[0,50): home → above_grasp
    for t in range(0, 50):
        ee[t] = lerp(home, above_grasp, t / 50)
        true_stage[t] = 0
    # grasp[50,80): 停在方块上方，抓取信号 1
    for t in range(50, 80):
        ee[t] = above_grasp.clone()
        true_stage[t] = 1
        grasp[t] = 1.0
    # lift[80,120): 抬升到 lift_top
    for t in range(80, 120):
        ee[t] = lerp(above_grasp, lift_top, (t - 80) / 40)
        true_stage[t] = 2
        grasp[t] = 1.0
    # move[120,160): 移到 above_stack（被夹方块随末端走，即 held 位置≈末端下压）
    for t in range(120, 160):
        ee[t] = lerp(lift_top, above_stack, (t - 120) / 40)
        true_stage[t] = 3
        grasp[t] = 1.0
    # stack[160,200): 停在底座上方
    for t in range(160, T):
        ee[t] = above_stack.clone()
        true_stage[t] = 4
        grasp[t] = 1.0

    # 被夹方块：grasp 后贴到末端位置下方，随末端一起走（真实世界被夹住的方块）
    held = torch.zeros(T, 3)
    held[:, 0] = GRASP_CUBE[0]
    held[:, 1] = GRASP_CUBE[1]
    held[:, 2] = TABLE_Z
    held[80:, 0] = ee[80:, 0]
    held[80:, 1] = ee[80:, 1]
    held[80:, 2] = ee[80:, 2] - 0.04  # 被夹住，略低于末端

    cube_positions = {
        "cube_2": held.clone(),                                    # 目标方块（被夹后随末端）
        "cube_1": STACK_CUBE.unsqueeze(0).expand(T, 3).clone(),    # 底座（静止）
        "cube_3": torch.tensor([-0.3, -0.3, TABLE_Z]).unsqueeze(0).expand(T, 3).clone(),
    }
    return cube_positions, true_stage, ee, grasp


def _features_of(cube_positions, ee, grasp, seed) -> torch.Tensor:
    features = extract_features(ee, cube_positions, grasp)
    features = features + torch.randn_like(features) * 0.01  # 轻微噪声
    return features


def test_unsupervised_recovers_stages():
    """合成 3 条五阶段轨迹，无监督聚类应恢复与真值高度一致的阶段。"""
    all_feat, all_true = [], []
    for i in range(3):
        cube_positions, true_stage, ee, grasp = _make_trajectory(i, seed=i)
        all_feat.append(_features_of(cube_positions, ee, grasp, i))
        all_true.append(true_stage)

    feat = torch.cat(all_feat, dim=0)
    true = torch.cat(all_true, dim=0)

    result = fit_unsupervised_centers(feat, n_stages=5)
    assign = result["assignments"]
    consistency = evaluate_consistency(assign, true)

    assert result["centers"].shape == (5, 5)
    assert assign.shape == true.shape
    assert consistency > 0.7, f"无监督阶段一致性过低：{consistency:.3f}"


def test_feature_subset_and_norm_reuse():
    """feature_cols 子集 + 归一化参数复用（assign_stages 对新数据可用）。"""
    cube_positions, true_stage, ee, grasp = _make_trajectory(0, seed=0)
    feat = _features_of(cube_positions, ee, grasp, 0)

    result = fit_unsupervised_centers(feat, n_stages=5, feature_cols=[F_GRASP_DIST, F_LIFT_HEIGHT, F_GRASP, F_STACK_DIST])
    assert result["centers"].shape == (5, 4)

    # 用学到的中心 + 归一化参数，对同一特征重新分配，应一致（须传相同的 feature_cols）
    re_assign = unsupervised.assign_stages(
        feat, result["centers"], norm=result["norm"],
        feature_cols=[F_GRASP_DIST, F_LIFT_HEIGHT, F_GRASP, F_STACK_DIST],
    )
    consistency = evaluate_consistency(re_assign, result["assignments"])
    assert consistency > 0.95, f"复用归一化重新分配不一致：{consistency:.3f}"


def test_requires_enough_samples():
    """样本数少于阶段数应抛错。"""
    feat = torch.randn(3, 5)
    try:
        fit_unsupervised_centers(feat, n_stages=5)
        assert False, "应抛 StageVLAError"
    except Exception as exc:
        assert "少于阶段数" in str(exc)


def test_progress_proxy_monotonic():
    """进度代理应随阶段递增（approach < ... < stack）。"""
    cube_positions, true_stage, ee, grasp = _make_trajectory(0, seed=0)
    feat = _features_of(cube_positions, ee, grasp, 0)
    proxy = unsupervised._progress_proxy(feat)
    # 每阶段平均代理应严格递增
    stage_means = [proxy[true_stage == s].mean().item() for s in range(5)]
    for a, b in zip(stage_means, stage_means[1:]):
        assert a < b, f"进度代理不单调：{stage_means}"
