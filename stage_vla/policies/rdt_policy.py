"""rdt_policy.py —— RDT-1B 策略封装（M2/M3 落地）。

M0 只注册构造器占位。M2 加载 ``RDTRunner``（third_party 拷贝引入）+ ``SiglipVisionTower``
+ 预编码 ``lang_embed.pt``；动作块用 ``ActionBlockQueue`` 消费。RDT 归一化边界
（8 维 state/action min/max、``num_views=3``）是硬约束，来自 ``config/rdt_normalization``。
"""

from __future__ import annotations

from ..core.config import Settings
from ..core.registry import register_policy
from .base import VLAPolicy


class ActionBlockQueue:
    """动作块队列 + 重规划（RDT 扩散模型输出 ``[horizon,8]`` 动作块）。M2 实现。"""

    def __init__(self, horizon: int = 64, replan_every: int = 8):
        self.horizon = horizon
        self.replan_every = replan_every

    def __len__(self) -> int:
        return 0


class RDT1BPolicy(VLAPolicy):
    """RDT-1B 策略（M2 实现 get_action）。"""

    def __init__(self, settings: Settings, **kwargs):
        self._settings = settings
        self._queue = ActionBlockQueue(
            horizon=settings.rdt_inference["horizon"],
            replan_every=settings.rdt_inference["replan_every"],
        )

    @property
    def action_dim(self) -> int:
        return 8

    def get_action(self, images, proprio):
        raise NotImplementedError("RDT1BPolicy.get_action 为 M2 里程碑实现。")

    def convert_action(self, action):
        raise NotImplementedError


@register_policy("rdt")
def build_rdt(settings: Settings, **kwargs) -> RDT1BPolicy:
    return RDT1BPolicy(settings, **kwargs)
