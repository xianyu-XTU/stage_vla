"""collect_demos.py —— 演示数据采集（M3 落地，补齐旧工程断链）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="采集演示数据 → outputs/demos/*.npz")
    parser.add_argument("--out_dir", type=str, default="outputs/demos")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    settings = load_settings()
    from stage_vla.data.demo_record import record_episode

    print(f"[collect_demos] 目标 {args.episodes} 段 → {args.out_dir}")
    record_episode(env=None, policy=None, instruction=settings.task["desc"], out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
