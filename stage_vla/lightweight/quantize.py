"""quantize.py —— 模型运行时量化（模块③，M3 落地）。

**INT8 不造假**：旧工程 ``quantize_runtime`` 文案声称支持 bits=4/8，但 ``bits=8`` 时
静默走 bf16 全精度（假量化）。本模块绝不复用该行为——``bits=8`` 直接抛
``NotImplementedError``，直到接入真实的 INT8 方案（optimum GPTQ/AWQ 或 torch.ao）。

- ``quantize(model, bits)``：运行时量化入口（M3 实现）
- ``report_memory(model)``：显存/参数量报告（旧版已验证，保留）
"""

from __future__ import annotations

import torch

from ..core.errors import StageVLAError


class QuantizationUnsupported(StageVLAError):
    """请求的量化位宽尚未接入真实实现。"""


def quantize(model: torch.nn.Module, bits: int) -> torch.nn.Module:
    """运行时量化模型权重。

    Args:
        model: 待量化模型
        bits: 量化位宽，仅支持 4（NF4）或 8

    Returns:
        量化后的模型

    Raises:
        QuantizationUnsupported: bits=8 尚未接入真实 INT8；bits=4 尚未实现。
    """
    if bits == 8:
        raise QuantizationUnsupported(
            "INT8 尚未接入真实实现（候选：optimum GPTQ/AWQ 或 torch.ao）。"
            "绝不复用旧工程 'INT8 静默走 bf16' 的假量化。"
        )
    if bits == 4:
        raise QuantizationUnsupported(
            "INT4（NF4）运行时量化为 M3 里程碑实现（bitsandbytes load_in_4bit）。"
        )
    raise QuantizationUnsupported(f"不支持的量化位宽：{bits}（仅 4 / 8）")


def report_memory(model: torch.nn.Module, tag: str = "model") -> dict:
    """报告模型参数量与张量显存占用（不含优化器）。

    Returns:
        {"params": int, "params_mb": float, "vram_mb": float, "device": str}
    """
    params = sum(p.numel() for p in model.parameters())
    vram_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    device = next(model.parameters()).device if any(model.parameters()) else "cpu"
    report = {
        "tag": tag,
        "params": params,
        "params_mb": round(params * 4 / 1e6, 2),
        "vram_mb": round(vram_bytes / 1e6, 2),
        "device": str(device),
    }
    return report
