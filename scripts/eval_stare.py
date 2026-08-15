"""eval_stare.py —— StARe-PPO 评估入口（M1 落地，**需 Isaac 环境**）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="StARe-PPO 评估")
    parser.add_argument("--load_run", type=str, required=True, help="运行目录前缀")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    from stage_vla.rl.eval import evaluate

    report = evaluate(
        settings,
        load_run=args.load_run,
        num_episodes=args.num_episodes,
        headless=args.headless,
    )
    print(f"[eval_stare] {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
