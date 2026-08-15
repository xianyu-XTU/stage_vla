"""语义分离测试：指令切分 / 动作关键词 / 颜色映射 / 兜底。"""

from __future__ import annotations

import pytest

from stage_vla.stages.semantic import SemanticSeparator


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
