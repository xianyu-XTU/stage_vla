"""core —— 基础设施：配置加载 / 日志 / 类型化异常 / 注册表。"""

from .encoding import ensure_utf8_output

ensure_utf8_output()

from .config import Settings, load_settings
from .errors import (
    ConfigError,
    DemoNotFoundError,
    PolicyNotFoundError,
    StageDetectionError,
    StageVLAError,
    TaskNotFoundError,
)

__all__ = [
    "ConfigError",
    "DemoNotFoundError",
    "PolicyNotFoundError",
    "Settings",
    "StageDetectionError",
    "StageVLAError",
    "TaskNotFoundError",
    "load_settings",
]
