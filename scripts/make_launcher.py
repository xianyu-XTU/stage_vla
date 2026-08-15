"""make_launcher.py —— 生成本机便捷启动脚本（gitignored，机器相关不入库）。

读 ``config.local.yaml``，生成 ``run.bat`` / ``train.bat`` 等便捷入口，
把 Isaac Lab 路径写进生成的 .bat（该文件被 .gitignore 忽略）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    isaaclab = settings.require_path("isaaclab")
    launcher = isaaclab / "isaaclab.bat"

    batch = f"""@echo off
rem 本机便捷启动（自动生成，勿手动编辑；由 scripts\\make_launcher.py 生成）
set ROOT={_ROOT}
echo [stage_vla] 仓库: %ROOT%
echo [stage_vla] 用 Isaac Lab 运行脚本: python tools\\run_isaaclab.py <script> [args]
"""
    out = _ROOT / "run.bat"
    out.write_text(batch, encoding="utf-8")
    print(f"已生成 {out}")
    print(f"Isaac Lab launcher: {launcher}")
    print("用法：python tools\\run_isaaclab.py scripts\\check_env.py --num_envs 4 --max_steps 5 --headless")
    return 0


if __name__ == "__main__":
    sys.exit(main())
