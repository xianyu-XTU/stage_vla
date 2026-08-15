"""factory.py —— 策略总出口（根治旧工程"兄弟脚本互相 import"）。

所有脚本只调用 :func:`build_policy`，按名查注册表构造策略实例；不要直接 import
具体策略模块。新策略用 :func:`~stage_vla.core.registry.register_policy` 注册。

当前注册的后端（均 M2/M3 落地，M0 仅接口）：
- ``openvla``      OpenVLA-7B（4-bit 自动检测）
- ``rdt``          RDT-1B（128 维状态映射 + ActionBlockQueue）
- ``vision_only``  去 LLM 的视觉动作策略
"""

from __future__ import annotations

from ..core.config import Settings
from ..core.errors import PolicyNotFoundError
from ..core.registry import POLICIES, register_policy
from .base import VLAPolicy


def build_policy(name: str, settings: Settings, **kwargs) -> VLAPolicy:
    """按名构造策略实例。

    Args:
        name: 策略后端名（openvla / rdt / vision_only）
        settings: 解析后的配置（策略需要 paths 等）
        kwargs: 透传给策略构造器

    Raises:
        PolicyNotFoundError: 策略未注册
    """
    try:
        builder = POLICIES.get(name)
    except PolicyNotFoundError:
        raise
    return builder(settings, **kwargs)


def list_policies() -> list[str]:
    """列出已注册的策略后端。"""
    return POLICIES.keys()


__all__ = ["build_policy", "list_policies", "register_policy"]
