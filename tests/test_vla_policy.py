"""VLA-as-policy 接口测试：稠密特征模式（不加载 3.2GB 视觉模型，纯 torch）。"""

from __future__ import annotations

import torch

from stage_vla.rl.vla_policy import DenseFeatureExtractor, VisionOnlyPPOPolicy


def _policy(features_dim: int = 8, stage_feedback_dim: int = 0) -> VisionOnlyPPOPolicy:
    return VisionOnlyPPOPolicy(
        features_dim=features_dim,
        action_dim=7,
        feature_extractor=DenseFeatureExtractor(features_dim),
        stage_feedback_dim=stage_feedback_dim,
        init_log_std=-1.0,
    )


def test_act_shapes_and_finite():
    """act → (action[B,7], log_prob[B], value[B]) 形状与有限值。"""
    policy = _policy(features_dim=8)
    obs = torch.randn(4, 8)
    action, log_prob, value = policy.act(obs)
    assert action.shape == (4, 7)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert bool(torch.isfinite(log_prob).all())
    assert bool(torch.isfinite(value).all())
    # 动作应服从先验（std≈exp(-1)≈0.37），均值接近 0
    assert bool((action.abs() < 2.0).all())


def test_evaluate_actions_matches_sample():
    """evaluate_actions 对采样动作应返回匹配的 log_prob 且熵为正。"""
    policy = _policy(features_dim=8)
    obs = torch.randn(4, 8)
    action, log_prob_sampled, _ = policy.act(obs)
    log_prob, entropy, value = policy.evaluate_actions(obs, action)
    torch.testing.assert_close(log_prob, log_prob_sampled, atol=1e-4, rtol=1e-4)
    assert bool((entropy > 0).all())
    assert value.shape == (4,)


def test_stage_feedback_conditioning():
    """阶段反馈作为条件输入（维度增加不影响接口）。"""
    policy = _policy(features_dim=8, stage_feedback_dim=5)
    obs = torch.randn(4, 8)
    feedback = torch.rand(4, 5)
    action, log_prob, value = policy.act(obs, stage_feedback=feedback)
    assert action.shape == (4, 7)
    assert bool(torch.isfinite(log_prob).all())


def test_trainable_parameters_only_heads():
    """冻结特征提取器：只有 actor/critic 头与 log_std 可训练。"""
    policy = _policy(features_dim=8)
    for p in policy.feature_extractor.parameters():
        p.requires_grad_(False)
    params = policy.trainable_parameters()
    names = {id(p) for p in params}
    # 头与 log_std 应可训练
    assert id(policy.log_std) in names
    assert id(policy.actor_head.net[0].weight) in names
    assert id(policy.critic_head.net[0].weight) in names
    # 参数数量少（头 ~2.1M 量级内）
    total = sum(p.numel() for p in params)
    assert total < 10_000_000


def test_dense_feature_extractor_flattens():
    fe = DenseFeatureExtractor(6)
    out = fe(torch.randn(3, 6))
    assert out.shape == (3, 6)
    # 1 维输入 → 加 batch
    assert fe(torch.randn(6)).shape == (1, 6)
