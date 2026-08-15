"""train_vla.py —— VLA-as-policy 融合训练入口（M2，自研 PPO 循环，**需 Isaac 环境**）。

用法::

    python tools\\run_isaaclab.py scripts\\train_vla.py --num_envs 16 --max_iterations 50 --headless
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
    parser = argparse.ArgumentParser(description="VLA-as-policy 融合训练（自研 PPO）")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--cam_res", type=int, default=128)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    instruction = args.instruction or settings.task["desc"]

    from stage_vla.rl.ppo_loop import train_vla_in_loop

    policy = train_vla_in_loop(
        settings,
        instruction=instruction,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        headless=args.headless,
        cam_res=args.cam_res,
    )
    print(f"[train_vla] 融合训练完成，策略就绪（可训练参数 {sum(p.numel() for p in policy.trainable_parameters())/1e6:.2f}M）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
