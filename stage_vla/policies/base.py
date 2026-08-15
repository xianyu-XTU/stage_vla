"""base.py —— VLA 策略抽象基类。

所有策略后端（OpenVLA / RDT / vision_only）实现同一接口，部署与训练代码按接口消费：

    VLAPolicy(ABC)
      ├── get_action(inputs)      # 给定观测/指令 → 原始动作
      ├── convert_action(action)  # 原始动作 → 环境动作（夹爪二进制化等）
      └── action_dim              # 输出动作维度
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLAPolicy(ABC):
    """视觉-语言-动作策略统一接口。"""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """输出动作维度（OpenVLA=7 / RDT=8 / vision_only=7）。"""
        raise NotImplementedError

    @abstractmethod
    def get_action(self, *args, **kwargs):
        """给定视觉/本体/指令输入，返回策略原始动作。"""
        raise NotImplementedError

    @abstractmethod
    def convert_action(self, action):
        """把策略原始动作转成环境可执行动作（如夹爪 0→开 / 1→夹）。"""
        raise NotImplementedError
