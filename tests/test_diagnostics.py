"""阶段诊断指标收集器测试（交接文档 16 节）。"""

from __future__ import annotations

import torch

from stage_vla.rl.diagnostics import StageMetrics


def test_stable_grasp_and_lift_detected():
    """20 步稳定抓取 + 抬离桌面 → stable_grasp_20step / lift_5cm 都命中。"""
    m = StageMetrics(stable_steps=20)
    m.reset()
    for step in range(30):
        physical = step >= 10                       # 第 10 步起开始抓
        cube_z = 0.02 + (0.01 if step < 20 else 0.08)  # 20 步后抬到 8cm
        m.record(
            stage=2 if physical and cube_z > 0.05 else 1,
            physical=physical,
            cube_z=cube_z,
            cube_vel=torch.tensor([0.0, 0.0, 0.0]),
            released=False,
            rb_frame=False,
        )
    m.end_episode(success=False)
    r = m.report()
    assert r["episodes"] == 1
    assert abs(r["stable_grasp_20step_rate"] - 1.0) < 1e-6
    assert abs(r["stable_grasp_10step_rate"] - 1.0) < 1e-6
    assert abs(r["lift_5cm_rate"] - 1.0) < 1e-6
    assert abs(r["task_success_rate"] - 0.0) < 1e-6


def test_short_grasp_not_stable():
    """只抓住 5 步 → 不算 stable_grasp_20step。"""
    m = StageMetrics(stable_steps=20)
    m.reset()
    for step in range(20):
        physical = step < 5
        m.record(1, physical=physical, cube_z=0.02, cube_vel=torch.zeros(3), released=False, rb_frame=False)
    m.end_episode(success=False)
    r = m.report()
    assert abs(r["stable_grasp_20step_rate"] - 0.0) < 1e-6


def test_success_and_stable_stack():
    """rb_frame 连续 25 步 + 释放 → stable_stack 与 success 都命中。"""
    m = StageMetrics(stable_steps=20)
    m.reset()
    for step in range(25):
        m.record(
            4, physical=False, cube_z=0.07, cube_vel=torch.zeros(3),
            released=True, rb_frame=True,   # 已放到蓝块上且释放、低速
        )
    m.end_episode(success=True)
    r = m.report()
    assert abs(r["stable_stack_10step_rate"] - 1.0) < 1e-6
    assert abs(r["stable_stack_20step_rate"] - 1.0) < 1e-6
    assert abs(r["task_success_rate"] - 1.0) < 1e-6
    assert abs(r["release_rate"] - 1.0) < 1e-6
