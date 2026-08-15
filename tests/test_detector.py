"""阶段检测器测试：逆序链 / 回归点 / 进度形状与单调性。"""

from __future__ import annotations

import torch

from stage_vla.stages.detector import StageDetector


def test_detect_inverse_chain(synth_stack_state):
    """4 个合成环境应判定为 approach/grasp/lift/stack。"""
    det = StageDetector()
    s = synth_stack_state
    stage = det.detect(s["ee"], s["cubes"], s["grasp"], s["stacked"])
    names = det.stage_names(stage)
    assert names == ["approach", "grasp", "lift", "stack"], names


def test_regression_is_stacked_independent_of_is_grasped():
    """旧 bug：用 is_grasped(stacked_signal) 判堆叠。必须独立。"""
    det = StageDetector()
    assert bool(det.is_stacked(torch.tensor(1.0)).item()) is True
    assert bool(det.is_grasped(torch.tensor(0.9)).item()) is True
    assert bool(det.is_grasped(torch.tensor(0.4)).item()) is False


def test_progress_shape_and_bounds(synth_stack_state):
    det = StageDetector()
    s = synth_stack_state
    prog = det.progress(s["ee"], s["cubes"], s["grasp"], s["stacked"])
    assert tuple(prog.shape) == (4, 5)
    assert bool((prog >= 0).all()) and bool((prog <= 1).all())


def test_progress_passed_stages_are_one(synth_stack_state):
    det = StageDetector()
    s = synth_stack_state
    prog = det.progress(s["ee"], s["cubes"], s["grasp"], s["stacked"])
    # env3 处于 stack，前面四阶段应全部完成
    assert bool((prog[3, :4] == 1).all())


def test_detect_from_settings(settings):
    """从配置构造的检测器应与默认阈值一致。"""
    det = StageDetector.from_settings(settings)
    assert det.stages == settings.stages
    assert det.thresholds["approach_dist"] == settings.thresholds["approach_dist"]
    assert det.cube_to_grasp == settings.task["cube_to_grasp"]
