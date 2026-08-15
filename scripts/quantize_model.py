"""quantize_model.py —— 模型量化入口（M3 落地；INT8 不造假）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="运行时量化")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=None)
    args = parser.parse_args()

    settings = load_settings()
    bits = args.bits or settings.lightweight["quant_bits"]

    from stage_vla.lightweight.quantize import quantize

    print(f"[quantize_model] 请求 {bits}-bit 量化")
    quantize(model=None, bits=bits)  # type: ignore[arg-type]  # M3 接入真实模型
    return 0


if __name__ == "__main__":
    sys.exit(main())
