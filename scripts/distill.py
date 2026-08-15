"""distill.py —— 知识蒸馏入口（M3 落地）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="知识蒸馏（teacher → student）")
    parser.add_argument("--demo_dir", type=str, default="outputs/demos")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    settings = load_settings()
    from stage_vla.lightweight.distill import distill

    distill(settings, demo_dir=args.demo_dir, num_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
