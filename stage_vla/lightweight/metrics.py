"""metrics.py —— 轻量化效果指标（量化与蒸馏共用）。

量化前 / 蒸馏后用它报告：参数量、峰值显存、推理延迟、每参数位宽（bits_per_param），
供实验对比与结题报告使用。
"""

from __future__ import annotations

import time

import torch


def params_count(model: torch.nn.Module) -> int:
    """模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters())


def vram_peak() -> float:
    """当前进程的峰值显存占用（MB）。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def infer_latency(fn, iters: int = 10, warmup: int = 2) -> float:
    """平均推理延迟（毫秒）。``fn`` 为无参可调用。"""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def bits_per_param(model: torch.nn.Module, quant_bits: int) -> float:
    """每参数平均位宽（总位数 / 参数量），量化收益指标。"""
    total = params_count(model)
    return (total * quant_bits) / max(total, 1)
