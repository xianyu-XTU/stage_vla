"""prismatic.py —— 共享 prismatic/OpenVLA 加载助手。

OpenVLA 依赖 prismatic（位于 ``paths.openvla_root``）。本模块集中处理：
- 把 ``openvla_root`` 加入 ``sys.path``（根治旧工程"兄弟脚本各自 sys.path.insert"）；
- 注册 ``AutoConfig.register("openvla", OpenVLAConfig)``；
- 幂等，供 OpenVLA / vision_only / 蒸馏等模块复用。
"""

from __future__ import annotations

import sys

from ..core.config import Settings

_registered = False


def ensure_prismatic_importable(settings: Settings) -> None:
    """把 OpenVLA 根加入 sys.path 并注册配置类（幂等）。"""
    global _registered
    if _registered:
        return
    openvla_root = str(settings.require_path("openvla_root"))
    if openvla_root not in sys.path:
        sys.path.insert(0, openvla_root)

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from transformers import AutoConfig

    AutoConfig.register("openvla", OpenVLAConfig)
    _registered = True
