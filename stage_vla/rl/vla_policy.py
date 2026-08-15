"""vla_policy.py —— VLA-as-policy（模块② 核心，M2 实现）。

申请书承诺"以 OpenVLA 为基础 + PPO 构建 StARe-PPO"，即把视觉语言策略当作 PPO 的
**actor**。rsl_rl 的 OnPolicyRunner 期望 MLP 风格 actor_critic，与 VLA 策略形态不匹配，
因此 M2 用**自研 PPO 循环**（见 ``ppo_loop.py``），策略实现本接口：

    VLAAsPolicy(nn.Module)
      ├── act(obs, stage_feedback) -> (action, log_prob, value)     # 采样动作 + 价值
      └── evaluate_actions(obs, action, stage_feedback)             # 评估（log_prob/entropy/value）
            -> (log_prob, entropy, value)

``VisionOnlyPPOPolicy`` 结构：
- ``feature_extractor``：把观测（图像/指令/状态）编码成特征 ``[B, F]``。M2a 用
  ``DenseFeatureExtractor``（obs 即特征，可纯 torch 单测）；M2b 换成视觉塔提取器。
- ``actor_head`` / ``critic_head``：小 MLP 输出动作均值 / 价值；``log_std`` 可学习。
- **8GB 纪律**：视觉塔冻结（前向 detach/bf16），只训练动作头与价值头（2.1M 参数量级）。
- ``stage_feedback`` 作为额外条件输入拼接进特征（阶段感知融合）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


# ============================================================================
# 特征提取器（可插拔：M2a 稠密特征 / M2b 视觉塔）
# ============================================================================
class FeatureExtractor(nn.Module):
    """把观测编码成策略特征 ``[B, F]``。"""

    @abstractmethod
    def forward(self, obs) -> torch.Tensor:
        raise NotImplementedError


class DenseFeatureExtractor(FeatureExtractor):
    """M2a 测试/稠密模式：观测本身即特征（无视觉塔）。"""

    def __init__(self, features_dim: int):
        super().__init__()
        self.features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x


# ============================================================================
# VLA 策略接口
# ============================================================================
class VLAAsPolicy(nn.Module, ABC):
    """可被 PPO 当作 actor 的 VLA 策略。"""

    #: 动作维度
    action_dim: int

    @abstractmethod
    def act(
        self,
        obs,
        stage_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作。

        Returns:
            (action [B, action_dim], log_prob [B], value [B])
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self,
        obs,
        action: torch.Tensor,
        stage_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作。

        Returns:
            (log_prob [B], entropy [B], value [B])
        """
        raise NotImplementedError


# ============================================================================
# VisionOnlyPPOPolicy（M2 实现）
# ============================================================================
class _MLPHead(nn.Module):
    """小 MLP：特征 → 输出。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionOnlyPPOPolicy(VLAAsPolicy):
    """冻结视觉特征 + 可训练动作/价值头 的 PPO 策略。

    Parameters
    ----------
    features_dim : 特征维度（视觉塔输出 F，或稠密模式下的输入维）
    action_dim : 动作维度（IK-Rel 7）
    feature_extractor : 观测 → [B, F] 的提取器；None 时 obs 即特征（稠密模式）
    stage_feedback_dim : 阶段反馈条件输入维度（0 = 不用）
    init_log_std : 初始对数标准差
    """

    action_dim = 7

    def __init__(
        self,
        features_dim: int,
        action_dim: int = 7,
        *,
        feature_extractor: FeatureExtractor | None = None,
        stage_feedback_dim: int = 0,
        init_log_std: float = -1.0,
        hidden: int = 256,
    ):
        super().__init__()
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.feature_extractor = feature_extractor
        self.stage_feedback_dim = stage_feedback_dim
        head_in = features_dim + stage_feedback_dim
        self.actor_head = _MLPHead(head_in, action_dim, hidden)
        self.critic_head = _MLPHead(head_in, 1, hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    # ------------------------------------------------------------------
    # 特征 + 阶段反馈 → 头输入
    # ------------------------------------------------------------------
    def _head_input(self, obs, stage_feedback: torch.Tensor | None) -> torch.Tensor:
        x = self.feature_extractor(obs) if self.feature_extractor is not None else torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if stage_feedback is not None:
            x = torch.cat([x, torch.as_tensor(stage_feedback, dtype=torch.float32)], dim=-1)
        return x

    def _distribution(self, x: torch.Tensor):
        mean = self.actor_head(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------
    def act(self, obs, stage_feedback=None):
        x = self._head_input(obs, stage_feedback)
        dist = self._distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, action, stage_feedback=None):
        x = self._head_input(obs, stage_feedback)
        dist = self._distribution(x)
        a = torch.as_tensor(action, dtype=torch.float32)
        log_prob = dist.log_prob(a).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return log_prob, entropy, value

    def trainable_parameters(self):
        """只训练动作/价值头与 log_std（冻结特征提取器）。"""
        params = list(self.actor_head.parameters()) + list(self.critic_head.parameters()) + [self.log_std]
        for p in params:
            p.requires_grad_(True)
        return params
