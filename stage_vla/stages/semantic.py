"""semantic.py —— 语义分离（模块① 入口，规则版）。

把语言指令（如 ``"pick up the red cube and place it on the blue cube"``）分离成语义
明确的子目标，再映射到子阶段序列，供阶段感知 RL 使用：

    语言指令 → 语义分离(子目标 + 目标物体) → 阶段序列 → 阶段计算器 → 稠密奖励 → PPO

本模块是**纯文本解析**，不依赖任何仿真/张量依赖，可无 Isaac Sim 单测。

实现说明：
- 用关键词规则解析动作（pick up / place on / stack 等）与目标物体（颜色 → 方块）。
- 这是"语义分离器"的规则版；``parse`` 接口固定，后续可替换为 LLM 解析（更灵活），
  调用方无需改动。

与旧工程差异：
- **去掉**旧 ``SemanticSeparator.detect_stage``（内部 import isaaclab 的几何判定）。
  语义 → 几何的接线放到 ``envs``（阶段检测器统一走 ``stages.calculator``），
  语义层保持纯文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 颜色 → 场景实体（与任务方块语义对应）
COLOR_TO_ENTITY = {
    "red": "cube_2",     # 红色方块（要抓的）
    "blue": "cube_1",    # 蓝色方块（放置底座）
    "green": "cube_3",   # 绿色方块（干扰）
}

# 动作关键词 → 阶段段
ACTION_KEYWORDS: dict[str, list[str]] = {
    "pick up": ["approach", "grasp", "lift"],
    "pick": ["approach", "grasp", "lift"],
    "grab": ["approach", "grasp", "lift"],
    "place on": ["move", "stack"],
    "place": ["move", "stack"],
    "stack": ["move", "stack"],
    "put on": ["move", "stack"],
}

# 兜底：未解析出任何子目标时的默认抓取目标
DEFAULT_GRASP_TARGET = "cube_2"
DEFAULT_STAGES = ("approach", "grasp", "lift", "move", "stack")


@dataclass
class SemanticSubGoal:
    """一个语义子目标（如"抓取红色方块"）。"""

    action: str
    target_entity: str
    stages: list[str]

    def __repr__(self) -> str:
        return f"<SubGoal {self.action} {self.target_entity} -> {self.stages}>"


@dataclass
class SemanticPlan:
    """语义分离的结果：一组有序子目标 + 完整阶段序列。"""

    sub_goals: list[SemanticSubGoal]
    stages: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 合并所有子目标的阶段，去重保序
        merged: list[str] = []
        for sg in self.sub_goals:
            for stage in sg.stages:
                if stage not in merged:
                    merged.append(stage)
        self.stages = merged

    def __repr__(self) -> str:
        return f"<SemanticPlan {self.sub_goals} stages={self.stages}>"


class SemanticSeparator:
    """语义分离器：把语言指令分离成语义子目标（规则版）。"""

    def __init__(
        self,
        color_to_entity: dict | None = None,
        action_keywords: dict | None = None,
        default_target: str = DEFAULT_GRASP_TARGET,
        default_stages: tuple[str, ...] = DEFAULT_STAGES,
    ):
        self.color_to_entity = color_to_entity or COLOR_TO_ENTITY
        self.action_keywords = action_keywords or ACTION_KEYWORDS
        self.default_target = default_target
        self.default_stages = list(default_stages)

    def parse(self, instruction: str) -> SemanticPlan:
        """把指令解析成语义子目标序列。"""
        text = instruction.lower()
        sub_goals: list[SemanticSubGoal] = []

        # 按连接词切分子句（and / then / 逗号）
        clauses = re.split(r"\band\b|\bthen\b|[,，]", text)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            sg = self._parse_clause(clause)
            if sg is not None:
                sub_goals.append(sg)

        if not sub_goals:
            # 兜底：整个指令作为一个子目标
            sg = self._parse_clause(text)
            sub_goals = [
                sg if sg is not None else SemanticSubGoal("default", self.default_target, self.default_stages)
            ]

        return SemanticPlan([sg for sg in sub_goals if sg is not None])

    def _parse_clause(self, clause: str) -> SemanticSubGoal | None:
        """解析单个子句：找动作关键词与目标物体。"""
        action = None
        stages: list[str] = []
        for kw, st in self.action_keywords.items():
            if kw in clause:
                action = kw
                stages = st
                break
        if action is None:
            return None

        target = None
        for color, entity in self.color_to_entity.items():
            if color in clause:
                target = entity
                break
        if target is None:
            target = self.default_target  # 兜底用默认抓取目标

        return SemanticSubGoal(action, target, stages)
