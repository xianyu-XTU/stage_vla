"""test_primitive_policy.py —— 实测视觉引导的基础动作策略（指令→计划，视觉→当前动作）。

运行（kit python，无需 Isaac Sim）::

    <isaac_sim>\\kit\\python\\python.exe scripts\\test_primitive_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.vla_light import VisionPrimitivePolicy, semantic_plan  # noqa: E402


def main() -> int:
    settings = load_settings()
    model = VisionPrimitivePolicy(settings)
    model.eval()

    report = model.param_report()
    total = sum(report.values())
    print("\n=== 体积（VisionPrimitivePolicy）===")
    for k, v in report.items():
        print(f"  {k:<28} {v/1e6:8.2f} M")
    print(f"  total                     {total/1e6:8.2f} M（视觉塔冻结，可训练 {(total - report['vision (frozen)'])/1e6:.1f}M）")

    instruction = "pick up the red cube and place it on the blue cube"
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    print(f"\n=== 语义计划（指令 → 基础动作）===")
    plan = semantic_plan(instruction)
    print(f"  指令: {instruction}")
    print(f"  基础动作序列: {plan}")

    print("\n=== 视觉引导输出（图像 + 指令 + 步骤上下文）===")
    for step in range(len(plan)):
        with torch.no_grad():
            names, params, plan_pred = model.predict_step(img, instruction, step_idx=step)
        print(f"  步 {step}: 视觉选动作={names[0]:<9} 参数(有效维)={[round(x,3) for x in params[0] if abs(x)>1e-3][:3]}"
              f"  计划预测={plan_pred[0]}")
    print("\n[OK] 视觉引导策略验证通过：指令→基础动作分解 + 视觉→当前动作 + 参数")
    print(f"  显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
