"""check_machine.py —— 本机环境探测。

打印解析后的配置与本机依赖情况：torch / CUDA / VRAM / isaaclab / openvla / rdt 路径存在性。
用于首次部署时确认机器就绪。

运行::

    python tools/check_machine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import check_machine, load_settings  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("本机环境探测")
    print("=" * 60)

    # Python / torch / CUDA
    print(f"python   : {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch    : {torch.__version__}")
        print(f"cuda     : {torch.version.cuda}")
        print(f"cuda 可用 : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu      : {torch.cuda.get_device_name(0)}")
            print(f"vram 总量 : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("torch    : 未安装（请用 Isaac Sim kit python 运行）")

    # 配置路径存在性
    print("\n--- 配置路径 ---")
    settings = load_settings()
    report = check_machine(verbose=True)
    missing_required = report.get("missing_required", [])
    missing_optional = [k for k in report.get("missing", []) if k not in missing_required]

    if missing_optional:
        print(f"\n可选路径缺失（VLA 线 M2/M3 才需要，不影响当前训练）：{missing_optional}")
    if missing_required:
        print(f"\n必需路径缺失：{missing_required}")
        print("提示：检查 config.local.yaml（参考 config/config.local.yaml.example）。")
        return 1
    print("\n必需路径就绪 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
