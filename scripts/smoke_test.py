"""smoke_test.py —— 第一版可跑原型（M0 交付，无需 Isaac Sim）。

验证 config → semantic → detector → rewards 全链路在合成张量上跑通：

  ① 默认配置无机器绝对路径（提交卫生约束）
  ② 指令解析出正确的阶段序列
  ③ 合成 4 环境状态 → 阶段索引 / 完成度 形状与数值正确
  ④ 势能塑形 + 阶段完成奖数值正确

运行（需含 torch 的 Python，如 Isaac Sim kit python）::

    python scripts/smoke_test.py

退出码 0 = 全部 PASS。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

# 把仓库根加入 sys.path，保证 `import stage_vla` 可用
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import DEFAULT_CONFIG_PATH, load_settings
from stage_vla.stages import SemanticSeparator, StageDetector
from stage_vla.stages.rewards import potential_shaping, stage_completion_reward

PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """断言并记录结果，失败直接 raise。"""
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    PASSED.append(name)
    print(f"  ✓ {name}{(' — ' + detail) if detail else ''}")


def main() -> int:
    print("=" * 64)
    print("stage_vla M0 冒烟测试：config → semantic → detector → rewards")
    print("=" * 64)

    # ---- ① 配置：默认配置必须机器无关（无盘符绝对路径） ----
    print("\n[1/4] 配置加载与卫生")
    default_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    drive_letters = re.findall(r"[A-Za-z]:\\", default_text)
    check("default.yaml 不含盘符绝对路径", not drive_letters, f"命中: {drive_letters}")

    settings = load_settings()
    check("config 加载成功", settings.stages == ["approach", "grasp", "lift", "move", "stack"])
    check(
        "reward_weights 与 stages 键对齐",
        all(s in settings.reward_weights for s in settings.stages),
        f"stages={settings.stages}",
    )
    check(
        "RDT 归一化 8 维约束",
        all(len(settings.rdt_normalization[k]) == 8 for k in ("state_min", "state_max", "action_min", "action_max")),
    )

    # ---- ② 语义分离 ----
    print("\n[2/4] 语义分离")
    instr = "pick up the red cube and place it on the blue cube"
    plan = SemanticSeparator().parse(instr)
    check("指令解析出五阶段", plan.stages == ["approach", "grasp", "lift", "move", "stack"], f"{plan.stages}")
    check("子目标数量为 2", len(plan.sub_goals) == 2, f"{plan.sub_goals}")

    # ---- ③ 阶段检测（合成 4 环境状态） ----
    print("\n[3/4] 阶段检测与完成度")
    det = StageDetector.from_settings(settings)
    N = 4
    ee = torch.tensor(
        [
            [0.5, 0.0, 0.15],  # env0: 远端 → approach
            [0.45, 0.0, 0.10],  # env1: 接近已抓住 → grasp
            [0.50, 0.0, 0.20],  # env2: 抓住并抬起 → lift
            [0.55, 0.0, 0.18],  # env3: 对齐并堆叠 → stack
        ],
        dtype=torch.float,
    )
    cubes = {
        "cube_1": torch.tensor([[0.55, 0.0, 0.02]] * N, dtype=torch.float),  # 底座（蓝）
        "cube_2": torch.tensor(  # 要抓（红）
            [[0.45, 0.0, 0.02], [0.45, 0.0, 0.04], [0.50, 0.0, 0.10], [0.55, 0.0, 0.08]],
            dtype=torch.float,
        ),
        "cube_3": torch.tensor([[0.5, -0.1, 0.02]] * N, dtype=torch.float),
    }
    grasp = torch.tensor([0.0, 1.0, 1.0, 1.0])
    stacked = torch.tensor([0.0, 0.0, 0.0, 1.0])

    stage = det.detect(ee, cubes, grasp, stacked)
    names = det.stage_names(stage)
    check("阶段判定", names == ["approach", "grasp", "lift", "stack"], f"{names}")

    # 回归点：is_stacked 独立于 is_grasped
    check("is_stacked 独立于 is_grasped", bool(det.is_stacked(torch.tensor(1.0))) and not bool(det.is_grasped(torch.tensor(0.4))))

    prog = det.progress(ee, cubes, grasp, stacked)
    check("progress 形状 [N, n_stages]", tuple(prog.shape) == (N, 5), f"{tuple(prog.shape)}")
    check("progress 边界 [0,1]", bool((prog >= 0).all()) and bool((prog <= 1).all()))
    check("progress 单调（已过阶段=1）", bool((prog[3, :4] == 1).all()), f"env3 前四阶段={prog[3, :4].tolist()}")

    # ---- ④ 奖励函数 ----
    print("\n[4/4] 稠密奖励")
    cur = torch.tensor([[0.8, 0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0]])
    prev = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0]])
    r = potential_shaping(cur, prev)
    check("势能塑形 进步>0 退步<0", r[0].item() > 0 and r[1].item() < 0, f"{r.tolist()}")

    stage_idx = torch.tensor([3, 2, 1, 4])
    prev_idx = torch.tensor([1, 0, 1, 3])
    bonus = stage_completion_reward(stage_idx, prev_idx, settings.reward_weights, settings.stages)
    w = settings.reward_weights
    expect = [w["lift"] + w["move"], w["grasp"] + w["lift"], 0.0, w["stack"]]
    check(
        "阶段完成奖励（跨级累加）",
        all(abs(a - b) < 1e-5 for a, b in zip(bonus.tolist(), expect)),
        f"{bonus.tolist()} expect {expect}",
    )

    print("\n" + "=" * 64)
    print(f"全部通过：{len(PASSED)} 项 PASS")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
