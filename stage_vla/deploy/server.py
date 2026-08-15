"""server.py —— VLA 推理 TCP 服务（M2 落地）。

8GB 显存下 OpenVLA-7B 与 Isaac 渲染不可共存，闭环部署用**双进程分离**：
本服务在独立进程加载 VLA 策略，仿真侧经 ``client.py`` 请求动作。

协议（M2 固化）：4 字节大端长度前缀 + pickle。请求 ``{instruction, frame}`` →
响应 ``np.ndarray[action_dim]``。

端口 / 策略名 / 模型路径全部走配置（``config/deploy.server`` + ``paths``），
**不再硬编码**（旧工程 ``vla_server.py`` 硬编码了本机绝对路径，属卫生问题，此处不重犯）。
"""

from __future__ import annotations

from ..core.config import Settings


def serve(settings: Settings, policy_name: str | None = None) -> None:
    """启动推理服务（M2 实现）。

    Args:
        settings: 解析后的配置
        policy_name: 策略后端名（缺省取 ``settings.rl.vla["vla_backend"]``）
    """
    raise NotImplementedError("VLA TCP 服务为 M2 里程碑实现。端口/策略/模型路径走配置。")
