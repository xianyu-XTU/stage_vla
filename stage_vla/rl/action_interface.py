"""action_interface.py —— 动作输出接口（模块② 核心预留，对应申请书"动作输出接口"）。

VLA 策略输出的是**任务空间动作**（OpenVLA 7 维 IK 增量、RDT 8 维关节 + 夹爪），而
Isaac 环境期望的是归一化动作张量。本接口收敛"VLA 动作 ↔ 环境动作"的双向映射，让
RL 训练 / 部署代码与具体策略解耦：

    ActionOutputInterface
      ├── to_env_action(vla_action) -> torch.Tensor   # 策略输出 → 环境可执行的归一化动作
      ├── from_env_state(obs) -> proprio             # 环境观测 → 策略期望的本体状态
      └── action_dim                                  # 环境动作维度

M0 只定义抽象与骨架，M2 融合训练时落地具体实现（IK 增量 / 关节位置 / 夹爪二进制化）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class ActionOutputInterface(ABC):
    """VLA 策略与 Isaac 环境之间的动作接口。"""

    #: 环境动作维度（Isaac env.action_space.shape[0]）
    action_dim: int

    @abstractmethod
    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """把策略原始输出转成环境可执行的归一化动作 ``[N, action_dim]``。

        Parameters
        ----------
        vla_action : VLA 策略的原始输出（7 维 IK 增量 / 8 维关节等）

        Returns
        -------
        环境 `step()` 可直接消费的 torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def from_env_state(self, obs: torch.Tensor) -> np.ndarray | torch.Tensor:
        """把环境观测转成策略期望的本体状态（RDT 需 128 维归一化状态等）。"""
        raise NotImplementedError


class IKRelActionInterface(ActionOutputInterface):
    """IK 增量动作接口（对齐 OpenVLA 7 维动作输出）。M2 实现。"""

    action_dim = 7

    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2 融合训练时实现：7 维 IK 增量 + 夹爪开合映射。")

    def from_env_state(self, obs: torch.Tensor) -> np.ndarray | torch.Tensor:
        raise NotImplementedError


class JointPosActionInterface(ActionOutputInterface):
    """关节位置动作接口（对齐 RDT 8 维动作输出）。M2 实现。"""

    action_dim = 8

    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2 融合训练时实现：8 维关节 + 夹爪反归一化。")

    def from_env_state(self, obs: torch.Tensor) -> np.ndarray | torch.Tensor:
        raise NotImplementedError
