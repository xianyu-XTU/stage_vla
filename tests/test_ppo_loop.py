"""PPO 循环纯逻辑测试（无 Isaac）：GAE 正确性 + 更新使损失下降。"""

from __future__ import annotations

import torch

from stage_vla.rl.ppo_loop import _compute_gae, _ppo_update
from stage_vla.rl.vla_policy import DenseFeatureExtractor, VisionOnlyPPOPolicy


def test_gae_discounted_sum():
    """GAE（γ=1, λ=1 退化为单步 TD 和）应对无折扣累计奖励正确。"""
    reward = torch.tensor([[1.0, 2.0], [0.0, 3.0], [5.0, 1.0]])  # [T=3, B=2]
    done = torch.zeros_like(reward)
    value = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # [T+1, B] 全 0
    rollout = {"reward": reward, "done": done, "value": value}
    adv, ret = _compute_gae(rollout, gamma=1.0, lam=1.0)
    # 回报 = 从该步起的累计奖励（λ=1 无折扣）
    assert torch.allclose(ret[0], reward.sum(dim=0))
    assert torch.allclose(ret[1], reward[1:].sum(dim=0))
    assert torch.allclose(ret[2], reward[2])


def test_gae_terminated_stops_bootstrap():
    """终止步后的 next-value 应被掩码（不引导到终止后）。"""
    reward = torch.tensor([[1.0]])          # [T=1, B=1]
    done = torch.tensor([[1.0]])            # 终止
    value = torch.tensor([[0.0], [10.0]])   # 终止后的 next-value 10，但应被忽略
    rollout = {"reward": reward, "done": done, "value": value}
    adv, ret = _compute_gae(rollout, gamma=0.99, lam=0.95)
    assert torch.allclose(ret[0], torch.tensor([1.0]))  # 不 + 0.99*10


def test_ppo_update_reduces_loss():
    """PPO 更新后策略损失应下降（overfit 一小批）。"""
    torch.manual_seed(0)
    policy = VisionOnlyPPOPolicy(
        features_dim=6, action_dim=7,
        feature_extractor=DenseFeatureExtractor(6), stage_feedback_dim=0,
        init_log_std=-1.0,
    )
    optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=1e-2)

    rollout = {
        "x": torch.randn(8, 6),
        "action": torch.randn(8, 7) * 0.5,
        "old_log_prob": torch.zeros(8),
    }
    advantage = torch.randn(8)
    returns = torch.randn(8) * 0.1

    before = _ppo_update(policy, optimizer, rollout, advantage, returns, ppo_epochs=10, mini_batches=2, clip=0.2)
    after = _ppo_update(policy, optimizer, rollout, advantage, returns, ppo_epochs=10, mini_batches=2, clip=0.2)
    # 价值头（MSE 回归）对同一批 returns 应持续下降（overfit 验证优化器工作）
    assert after["value_loss"] < before["value_loss"] - 1e-4
    # 策略损失（-clip surrogate）应更负 = 代理收益更大
    assert after["policy_loss"] < before["policy_loss"] - 1e-4


def test_ppo_update_shape_and_finite():
    policy = VisionOnlyPPOPolicy(
        features_dim=6, action_dim=7,
        feature_extractor=DenseFeatureExtractor(6), stage_feedback_dim=3,
    )
    optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=1e-3)
    rollout = {
        "x": torch.randn(16, 9),          # 特征 6 + 阶段反馈 3
        "action": torch.randn(16, 7),
        "old_log_prob": torch.randn(16),
    }
    adv, ret = torch.randn(16), torch.randn(16)
    losses = _ppo_update(policy, optimizer, rollout, adv, ret, ppo_epochs=2, mini_batches=4, clip=0.2)
    for v in losses.values():
        assert torch.isfinite(torch.as_tensor(v))
