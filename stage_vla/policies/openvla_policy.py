"""openvla_policy.py —— OpenVLA-7B 策略封装（M2/M3 落地）。

M0 只注册构造器占位：可被 :func:`~stage_vla.policies.factory.build_policy` 查到，
但 ``get_action`` 抛指引性错误。M2 用 transformers + ``prismatic.extern.hf`` 加载
``OpenVLAForActionPrediction``；``use_4bit="auto"`` 时 <12GB 显存自动 NF4（≈4.1GB）。
"""

from __future__ import annotations

from ..core.config import Settings
from ..core.errors import PolicyNotFoundError
from ..core.registry import register_policy
from .base import VLAPolicy


class OpenVLAPolicy(VLAPolicy):
    """OpenVLA-7B 策略（M2 实现 get_action）。"""

    def __init__(self, settings: Settings, use_4bit="auto", **kwargs):
        self._settings = settings
        self._use_4bit = use_4bit
        self._model = None

    @property
    def action_dim(self) -> int:
        return 7

    def get_action(self, image, instruction):
        raise NotImplementedError(
            "OpenVLAPolicy.get_action 为 M2 里程碑实现。"
            "8GB 显存下 OpenVLA-7B 与渲染不可共存，融合走 record/replay 或 TCP 分离进程。"
        )

    def convert_action(self, action):
        raise NotImplementedError


@register_policy("openvla")
def build_openvla(settings: Settings, **kwargs) -> OpenVLAPolicy:
    return OpenVLAPolicy(settings, **kwargs)
