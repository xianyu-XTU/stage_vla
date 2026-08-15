"""encoding.py —— UTF-8 输出保障（Windows GBK 控制台兼容）。

Windows 中文控制台默认 cp936（GBK），无法编码 ``✓``/``中文`` 等字符，导致
``UnicodeEncodeError`` 或乱码。本模块在入口处重配 stdout/stderr 为 UTF-8。
"""

from __future__ import annotations

import sys

_APPLIED = False


def ensure_utf8_output() -> None:
    """把 stdout / stderr 重配为 UTF-8（幂等）。"""
    global _APPLIED
    if _APPLIED:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    _APPLIED = True
