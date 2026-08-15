"""统一日志：console + 可选文件轮转。

用法::

    from stage_vla.core.logging import get_logger
    logger = get_logger("stages.rewards")
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT_LOGGER_NAME = "stage_vla"
_configured = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    """配置根 logger：console handler + 可选 RotatingFileHandler。

    幂等：重复调用只更新日志目录（如需），不重复添加 handler。
    """
    global _configured
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # 移除旧的同名文件 handler 再重建，避免重复写
        for handler in list(logger.handlers):
            if isinstance(handler, RotatingFileHandler):
                logger.removeHandler(handler)
        file_handler = RotatingFileHandler(
            log_dir / "stage_vla.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """返回以 ``stage_vla.<name>`` 命名的 logger（若未配置过先 setup console）。"""
    if not _configured:
        setup_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
