"""client.py —— 仿真侧推理客户端（配 server.py，M2 落地）。"""

from __future__ import annotations

from ..core.config import Settings


def request_action(settings: Settings, instruction: str, frame, timeout: float = 5.0):
    """向推理服务请求动作（M2 实现）。

    Args:
        settings: 解析后的配置（含 deploy.server 端口）
        instruction: 语言指令
        frame: 相机帧（np.ndarray）
        timeout: 秒
    """
    raise NotImplementedError("推理客户端为 M2 里程碑实现。")
