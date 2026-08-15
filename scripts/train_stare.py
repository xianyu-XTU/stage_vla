"""train_stare.py —— StARe-PPO 训练入口（v1 状态版，**需 Isaac 环境**）。

用法::

    python tools\\run_isaaclab.py scripts\\train_stare.py --num_envs 32 --max_iterations 50 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="StARe-PPO v1 状态版训练")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--instruction", type=str, default=None,
                        help="语言指令（语义分离入口，缺省用 config task.desc）")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    instruction = args.instruction or settings.task["desc"]

    from stage_vla.rl.stare_ppo import train_stare

    runner = train_stare(
        settings,
        instruction=instruction,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        headless=args.headless,
    )
    print(f"[train_stare] 训练完成：{runner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
