"""VLA 策略层：统一接口 + 按名构造。

``import stage_vla.policies`` 时自动导入三个后端模块触发注册表登记，
之后脚本只调 :func:`stage_vla.policies.build_policy`。
"""

from . import openvla_policy, rdt_policy, vision_only_policy  # noqa: F401  触发注册
from .base import VLAPolicy
from .factory import build_policy, list_policies

__all__ = ["VLAPolicy", "build_policy", "list_policies"]
