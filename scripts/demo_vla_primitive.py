"""demo_vla_primitive.py —— 实测：输入指令 + 视觉画面 → 输出机械臂动作（基础动作 + 计划）。

用训练好的模型（outputs/primitives_model.pt），喂真实仿真图像 + 指令，
展示模型根据视觉判断当前基础动作、根据指令分解计划。

运行（kit python，无需 Isaac Sim）::

    <isaac_sim>\\kit\\python\\python.exe scripts\\demo_vla_primitive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.vla_light import VisionPrimitivePolicy  # noqa: E402
from stage_vla.vla_light.primitives import PRIMITIVE_NAMES  # noqa: E402

INSTRUCTIONS = [
    "pick up the red cube and place it on the blue cube",
    "just grab the green cube",
    "stack the red cube on the blue cube",
]


def main() -> int:
    settings = load_settings()
    model = VisionPrimitivePolicy(settings)
    ckpt = _ROOT / "outputs" / "primitives_model.pt"
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False))
    model.eval()
    print(f"[demo] 已加载训练模型 {ckpt}")

    # 加载真实仿真图像（脚本化阶段位姿数据）
    data = torch.load(str(_ROOT / "outputs" / "primitive_data.pt"), map_location="cpu", weights_only=False)
    imgs, stages = data["images"], data["stages"]

    # 每阶段取一张图像，用不同指令测试
    print("\n=== 指令 + 视觉画面 → 输出机械臂动作 ===")
    for instr in INSTRUCTIONS:
        print(f"\n指令: {instr}")
        for st in [0, 2, 4]:                      # approach / lift / stack 各取一张
            idx = int((stages == st).nonzero()[0][0])
            img = imgs[idx]
            with torch.no_grad():
                names, params, plan = model.predict_step(img, instr, step_idx=0)
            hit = "✓" if names[0] == PRIMITIVE_NAMES[st] else "✗"
            print(f"  [真实阶段={PRIMITIVE_NAMES[st]}] 视觉判定={names[0]} {hit}"
                  f"  | 计划={plan}")

    print("\n=== 参数（当前未训练，仅为结构演示）===")
    img = imgs[0]
    names, params, plan = model.predict_step(img, INSTRUCTIONS[0], step_idx=0)
    print(f"  动作={names[0]} 参数(有效维)={[round(p,3) for p in params[0] if abs(p)>1e-3][:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
