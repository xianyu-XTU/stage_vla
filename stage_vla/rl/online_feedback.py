"""online_feedback.py —— 在线反馈循环（模块② 预留，对应申请书"在线反馈机制"）。

让 VLA 策略能根据环境实时变化（物体位置偏移、碰撞、光照变化等）动态调整动作：

    OnlineFeedback
      ├── on_step(obs, action, reward, stage, done)   # 每一步被训练/部署循环回调
      └── feedback() -> dict                           # 供策略消费的反馈上下文

M0 提供基类与 ``NoopFeedback``（占位）；M2 实现 ``StageFeedback``：把阶段信号回灌
VLA 上下文 / conditioning，实现"融合训练闭环"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OnlineFeedback(ABC):
    """在线反馈循环基类。"""

    @abstractmethod
    def on_step(self, obs, action, reward, stage, done) -> None:
        """训练/部署循环每步回调。子类在此累积反馈信号。"""
        raise NotImplementedError

    @abstractmethod
    def feedback(self) -> dict:
        """返回当前反馈上下文（供策略 condition / 奖励塑形使用）。"""
        raise NotImplementedError


class NoopFeedback(OnlineFeedback):
    """空反馈：不产生任何上下文。M2 融合前的默认占位。"""

    def on_step(self, obs, action, reward, stage, done) -> None:
        return None

    def feedback(self) -> dict:
        return {}


class StageFeedback(OnlineFeedback):
    """阶段反馈：把当前阶段信号累积为上下文，供 VLA 条件输入（M2 实现）。

    在 PPO 循环每步调用 :meth:`on_step`（传入 ``stage`` 为 ``[B]`` 阶段索引张量），
    :meth:`feedback` 返回 ``[B, n_stages]`` 的 one-hot 阶段条件输入，拼接进策略特征。
    """

    def __init__(self, stages: list[str]):
        self._stages = list(stages)
        self._n_stages = len(stages)
        self._current: torch.Tensor | None = None

    def on_step(self, obs, action, reward, stage, done) -> None:
        self._current = torch.as_tensor(stage)

    def feedback(self) -> torch.Tensor:
        """返回 ``[B, n_stages]`` 当前阶段 one-hot（尚未收到任何 step 时为 None）。"""
        if self._current is None:
            return None
        cur = self._current
        if cur.dim() == 0:
            cur = cur.unsqueeze(0)
        B = cur.shape[0]
        onehot = torch.zeros(B, self._n_stages, dtype=torch.float32)
        onehot[torch.arange(B), cur.long().clamp(0, self._n_stages - 1)] = 1.0
        return onehot

    def reset(self) -> None:
        self._current = None
