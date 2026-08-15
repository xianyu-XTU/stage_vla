"""vision_only_policy.py —— 去 LLM 的视觉动作策略（M2/M3 落地）。

M0 只注册构造器占位。M2 从 ``outputs/openvla_vision_only`` 重建 vision_backbone +
projector + 随机初始化动作头（动作头需演示数据训练才会抓取，见 M3 演示链）。
"""

from __future__ import annotations

from ..core.config import Settings
from ..core.registry import register_policy
from .base import VLAPolicy


class VisionOnlyPolicy(VLAPolicy):
    """视觉动作策略（M2 实现 get_action）。"""

    def __init__(self, settings: Settings, **kwargs):
        self._settings = settings

    @property
    def action_dim(self) -> int:
        return 7

    def get_action(self, image, instruction_embed):
        raise NotImplementedError("VisionOnlyPolicy.get_action 为 M2 里程碑实现。")

    def convert_action(self, action):
        raise NotImplementedError


@register_policy("vision_only")
def build_vision_only(settings: Settings, **kwargs) -> VisionOnlyPolicy:
    return VisionOnlyPolicy(settings, **kwargs)
