"""train_stare.py —— StARe-PPO 训练入口（v1 状态版，**需 Isaac 环境**）。

用法::

    python tools\\run_isaaclab.py scripts\\train_stare.py --num_envs 64 --max_iterations 200 --headless
    python tools\\run_isaaclab.py scripts\\train_stare.py --use_official --num_envs 64 --max_iterations 200 --headless
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def run_official_baseline(settings, num_envs: int, max_iterations: int, headless: bool = True) -> int:
    """对照实验：官方训练脚本（无阶段感知塑形，用任务默认奖励）。"""
    isaaclab_dir = settings.require_path("isaaclab")
    train_script = isaaclab_dir / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    cmd = [
        str(isaaclab_dir / "isaaclab.bat"), "-p", str(train_script),
        "--task", settings.task["id_state"],
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
    ]
    if headless:
        cmd.append("--headless")
    print(f"[train] 官方基线命令: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="StARe-PPO v1 状态版训练 / 官方基线对照")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--instruction", type=str, default=None,
                        help="语言指令（语义分离入口，缺省用 config task.desc）")
    parser.add_argument("--use_official", action="store_true",
                        help="走官方 train.py（无阶段感知奖励，依赖旧源码注册的 agent 配置），作对照实验")
    parser.add_argument("--baseline", action="store_true",
                        help="同 v2 管线但**不注入阶段奖励**（保留任务默认奖励，PPO 配置与组 A 全同），公平对照")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    instruction = args.instruction or settings.task["desc"]

    if args.use_official:
        return run_official_baseline(settings, args.num_envs, args.max_iterations, args.headless)

    from stage_vla.rl.stare_ppo import train_stare

    runner = train_stare(
        settings,
        instruction=instruction,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        headless=args.headless,
        stage_rewards=not args.baseline,
    )
    print(f"[train_stare] 训练完成：{runner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
