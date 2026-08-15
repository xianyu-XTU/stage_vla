"""语义分离测试：指令切分 / 动作关键词 / 颜色映射 / 兜底。"""

from __future__ import annotations

import pytest

from stage_vla.stages.semantic import (
    SemanticSeparator,
    filter_stage_weights,
    plan_targets,
)


@pytest.fixture
def sep() -> SemanticSeparator:
    return SemanticSeparator()


def test_parse_full_stack_instruction(sep):
    plan = sep.parse("pick up the red cube and place it on the blue cube")
    assert plan.stages == ["approach", "grasp", "lift", "move", "stack"]
    assert len(plan.sub_goals) == 2
    assert plan.sub_goals[0].target_entity == "cube_2"  # red → 要抓
    assert plan.sub_goals[1].target_entity == "cube_1"  # blue → 底座


def test_parse_then_connective(sep):
    plan = sep.parse("grab the green cube then put on the red cube")
    assert plan.sub_goals[0].stages == ["approach", "grasp", "lift"]
    assert plan.sub_goals[1].stages == ["move", "stack"]
    assert plan.stages == ["approach", "grasp", "lift", "move", "stack"]


def test_parse_chinese_comma(sep):
    plan = sep.parse("pick up the red cube, stack on the blue cube")
    assert plan.stages == ["approach", "grasp", "lift", "move", "stack"]


def test_parse_unknown_instruction_fallback(sep):
    plan = sep.parse("hello robot do your best")
    assert plan.sub_goals, "兜底应产出至少一个子目标"
    assert plan.sub_goals[0].action == "default"
    assert plan.stages, "兜底阶段序列非空"


def test_parse_color_mapping_case_insensitive(sep):
    plan = sep.parse("PICK UP THE RED CUBE")
    assert plan.sub_goals[0].target_entity == "cube_2"
    assert plan.sub_goals[0].stages == ["approach", "grasp", "lift"]


# ---- 语义分离驱动训练（M1 剩余项）----

def test_plan_targets_full_instruction(sep):
    """完整指令 → 抓 cube_2 放 cube_1，五阶段全活动。"""
    plan = sep.parse("pick up the red cube and place it on the blue cube")
    grasp, stack, active = plan_targets(plan)
    assert grasp == "cube_2"
    assert stack == "cube_1"
    assert active == ["approach", "grasp", "lift", "move", "stack"]


def test_plan_targets_pick_only(sep):
    """只抓不放 → 目标 cube_3，活动阶段只到 lift（move/stack 不奖励）。"""
    plan = sep.parse("just grab the green cube")
    grasp, stack, active = plan_targets(plan)
    assert grasp == "cube_3"
    assert stack == "cube_1"  # 无放置子目标 → 默认底座
    assert active == ["approach", "grasp", "lift"]


def test_filter_stage_weights_zeros_unplanned():
    weights = {"action_penalty": -0.01, "progress_shaping": 1.0,
               "approach": 0.0, "grasp": 2.0, "lift": 1.0, "move": 0.5, "stack": 10.0}
    stages = ["approach", "grasp", "lift", "move", "stack"]
    filtered = filter_stage_weights(weights, stages, ["approach", "grasp", "lift"])
    assert filtered["grasp"] == 2.0       # 活动阶段保留
    assert filtered["lift"] == 1.0
    assert filtered["move"] == 0.0        # 未活动阶段置 0
    assert filtered["stack"] == 0.0
    assert filtered["action_penalty"] == -0.01  # 全局项不受影响
    # None = 全部活动
    assert filter_stage_weights(weights, stages, None) == weights
