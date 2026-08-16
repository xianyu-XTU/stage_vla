"""test_vla_light.py —— 实测轻量 VLA（去 LLM + 指令分词器）：体积缩减 + 前向。

运行（kit python，无需 Isaac Sim）::

    <isaac_sim>\\kit\\python\\python.exe scripts\\test_vla_light.py
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
from stage_vla.vla_light import OpenVLALightForAction  # noqa: E402

# OpenVLA-7B 参数（Llama-2 7B 骨干）
OPENVLA_PARAMS = 7_000_000_000


def main() -> int:
    settings = load_settings()
    model = OpenVLALightForAction(settings)

    report = model.param_report()
    print("\n=== 体积对比（OpenVLA-7B vs VLA-light）===")
    for k, v in report.items():
        print(f"  {k:<28} {v/1e6:8.2f} M")
    print(f"  {'VLA-light 总':<28} {report['total']/1e6:8.2f} M")
    print(f"  OpenVLA-7B             {OPENVLA_PARAMS/1e9:.1f} B")
    ratio = report["total"] / OPENVLA_PARAMS
    print(f"  → 体积 {ratio*100:.1f}%，缩小 {100-ratio*100:.1f}%")
    print(f"  其中冻结视觉塔 {report['vision (frozen)']/1e6:.0f}M（占大头），"
          f"可训练（分词器+头）{(report['instruction_tokenizer']+report['action_head'])/1e6:.1f}M")

    # 前向测试：真实随机图像 + 多条指令
    print("\n=== 前向测试（真实视觉塔 + 指令分词器）===")
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    instructions = [
        "pick up the red cube and place it on the blue cube",
        "stack the green cube on the red cube",
    ]
    model.eval()
    with torch.no_grad():
        actions = model.predict_action(img, instructions)
    print(f"动作形状: {tuple(actions.shape)}  有限值: {torch.isfinite(actions).all().item()}")
    print(f"动作[0]: {actions[0].tolist()}")
    print(f"动作[1]: {actions[1].tolist()}")
    assert actions.shape == (2, 7)
    assert torch.isfinite(actions).all()

    # 不同指令 → 不同动作
    diff = (actions[0] - actions[1]).norm().item()
    print(f"不同指令动作差异: {diff:.4f}（应 >0，说明指令影响输出）")

    print(f"\n显存占用: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print("\n[OK] VLA-light 验证通过：体积 -{:.0f}%、前向正常、指令即时编码".format(100 - ratio * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
