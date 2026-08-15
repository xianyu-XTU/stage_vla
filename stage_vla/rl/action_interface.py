"""action_interface.py —— 动作输出接口（模块② 核心，对应申请书"动作输出接口"）。

VLA 策略输出的是**任务空间动作**（OpenVLA/vision_only 7 维 IK 增量 + 夹爪；RDT 8 维
关节 + 夹爪），Isaac 环境期望归一化动作张量。本接口收敛"VLA 动作 ↔ 环境动作"双向映射：

    ActionOutputInterface
      ├── to_env_action(vla_action) -> torch.Tensor   # 策略输出 → 环境可执行动作
      ├── from_env_state(obs) -> np.ndarray|None      # 环境观测 → 策略本体状态（VLA 通常不需）
      └── action_dim                                  # 环境动作维度

M2 实现：
- ``IKRelActionInterface``  对齐 OpenVLA/vision_only 7 维（6 末端增量 + 1 夹爪）
- ``JointPosActionInterface`` 对齐 RDT 8 维（7 关节 + 1 夹爪，经 rdt_normalization
  归一化到 128 维统一动作空间）

128 维归一化逻辑抽成**纯函数**（``encode_state_128`` / ``decode_action_128``），
便于无 Isaac 单测（M2 验收：to_env_action → env.step → from_env_state 往返一致）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

# RDT 统一动作空间维度（7 关节 + 夹爪的归一化域，其余为 0）
RDT_STATE_DIM = 128
RDT_CONTROL_DIM = 8


# ============================================================================
# 纯函数：RDT 128 维 <-> 8 维归一化映射（可独立单测）
# ============================================================================
def encode_state_128(state_8: torch.Tensor, state_min, state_max) -> torch.Tensor:
    """把 8 维关节状态（7 关节 + 夹爪）归一化到 [-1,1] 并填入 128 维域。

    Args:
        state_8: [..., 8] 原始关节状态
        state_min / state_max: [8] rdt_normalization 边界

    Returns:
        [..., 128]（前 8 维归一化，其余 0）
    """
    norm = 2.0 * (state_8 - state_min) / (state_max - state_min) - 1.0
    out = torch.zeros(*state_8.shape[:-1], RDT_STATE_DIM, device=state_8.device, dtype=state_8.dtype)
    out[..., :RDT_CONTROL_DIM] = norm
    return out


def decode_action_128(action_128: torch.Tensor, action_min, action_max) -> torch.Tensor:
    """把 128 维动作域的前 8 维反归一化回 8 维关节动作（7 关节 + 夹爪）。

    Args:
        action_128: [..., 128] RDT 统一动作空间
        action_min / action_max: [8] rdt_normalization 边界

    Returns:
        [..., 8]
    """
    norm = action_128[..., :RDT_CONTROL_DIM]
    return (norm + 1.0) * (action_max - action_min) / 2.0 + action_min


# ============================================================================
# 动作接口
# ============================================================================
class ActionOutputInterface(ABC):
    """VLA 策略与 Isaac 环境之间的动作接口。"""

    #: 环境动作维度（Isaac env.action_space.shape[0]）
    action_dim: int

    @abstractmethod
    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """把策略原始输出转成环境可执行的归一化动作 ``[N, action_dim]``。"""
        raise NotImplementedError

    @abstractmethod
    def from_env_state(self, obs) -> np.ndarray | torch.Tensor | None:
        """把环境观测转成策略期望的本体状态（VLA 通常不需要，返回 None）。"""
        raise NotImplementedError


class IKRelActionInterface(ActionOutputInterface):
    """IK 相对动作接口（对齐 OpenVLA / vision_only 7 维：6 末端增量 + 1 Binary 夹爪）。"""

    action_dim = 7

    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """[N,7] → [N,7]：6 维末端增量**夹到 [-1,1]**（IK-Rel 环境期望归一化动作，
        未夹的随机策略输出会让物理仿真爆掉——M2 实测 CUDA device assert）；
        夹爪符号 → 二进制 ±1（匹配 BinaryJointPositionActionCfg）。"""
        a = torch.as_tensor(vla_action, dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        out = a.clone()
        out[:, :6] = a[:, :6].clamp(min=-1.0, max=1.0)   # IK 增量限幅
        out[:, 6] = torch.sign(a[:, 6]).clamp(min=-1.0, max=1.0)  # 夹爪：>0 开 / <0 夹
        return out

    def from_env_state(self, obs) -> None:
        """OpenVLA/vision_only 是视觉+语言驱动，不需要本体状态。"""
        return None


class JointPosActionInterface(ActionOutputInterface):
    """关节位置动作接口（对齐 RDT 8 维：7 关节 + 1 夹爪，经 rdt_normalization 映射 128 维）。"""

    action_dim = 8

    def __init__(self, state_min, state_max, action_min, action_max):
        self._state_min = torch.as_tensor(state_min, dtype=torch.float32)
        self._state_max = torch.as_tensor(state_max, dtype=torch.float32)
        self._action_min = torch.as_tensor(action_min, dtype=torch.float32)
        self._action_max = torch.as_tensor(action_max, dtype=torch.float32)

    @classmethod
    def from_settings(cls, settings) -> "JointPosActionInterface":
        norm = settings.rdt_normalization
        return cls(norm["state_min"], norm["state_max"], norm["action_min"], norm["action_max"])

    def to_env_action(self, vla_action: np.ndarray | torch.Tensor) -> torch.Tensor:
        """[N,128] → [N,8]：反归一化 RDT 动作域的前 8 维。"""
        a = torch.as_tensor(vla_action, dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return decode_action_128(a, self._action_min, self._action_max)

    def from_env_state(self, obs: torch.Tensor) -> torch.Tensor:
        """[N,8] 关节状态 → [N,128] 归一化本体状态。"""
        s = torch.as_tensor(obs, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return encode_state_128(s, self._state_min, self._state_max)
