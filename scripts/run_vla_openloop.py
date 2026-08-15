"""run_vla_openloop.py —— VLA 开环推理入口（M2 落地，8GB 显存 record/replay 模式）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA 开环推理（record/replay/offline）")
    parser.add_argument("--mode", choices=["record", "replay", "offline"], default="record")
    parser.add_argument("--policy", type=str, default=None, help="策略后端（缺省取 config rl.vla.vla_backend）")
    parser.add_argument("--instruction", type=str, default=None)
    args = parser.parse_args()

    settings = load_settings()
    instruction = args.instruction or settings.task["desc"]
    policy_name = args.policy or settings.rl["vla"]["vla_backend"]

    from stage_vla.policies import build_policy

    policy = build_policy(policy_name, settings)
    print(f"[run_vla_openloop] mode={args.mode} policy={policy_name} instruction={instruction!r}")
    print("提示：get_action 为 M2 里程碑实现，当前策略返回 NotImplementedError。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
