"""run_isaaclab.py —— 用 Isaac Lab 的 isaaclab.bat 运行脚本（机器无关）。

旧工程把本机 Isaac Lab 绝对路径硬编码进 ``isaaclab_stage_vla.bat``；本工具改为读
``config.local.yaml`` 的 ``paths.isaaclab``，任何机器零改动即可运行。

用法::

    python tools\\run_isaaclab.py scripts\\train_stare.py --num_envs 32 --max_iterations 50 --headless

会执行::

    <isaaclab>/isaaclab.bat -p <repo>/scripts/train_stare.py --num_envs 32 ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    script = Path(argv[0])
    if not script.is_absolute():
        script = _ROOT / script
    if not script.is_file():
        print(f"[错误] 脚本不存在：{script}")
        return 2

    settings = load_settings()
    isaaclab_dir = settings.require_path("isaaclab")
    launcher = isaaclab_dir / "isaaclab.bat"
    if not launcher.is_file():
        print(f"[错误] 未找到 isaaclab.bat：{launcher}")
        print("请检查 config.local.yaml 的 paths.isaaclab。")
        return 1

    cmd = [str(launcher), "-p", str(script), *argv[1:]]
    print("执行：", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_ROOT), env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
