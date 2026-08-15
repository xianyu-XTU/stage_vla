"""finetune_lora.py —— QLoRA 低秩微调入口（M3 落地）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA 微调 OpenVLA（只存 adapter）")
    parser.add_argument("--demo_dir", type=str, default="outputs/demos")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs/checkpoints/lora")
    args = parser.parse_args()

    settings = load_settings()
    from stage_vla.lightweight.lora import finetune  # type: ignore[attr-defined]  # M3 落地

    finetune(settings, demo_dir=args.demo_dir, epochs=args.epochs, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
