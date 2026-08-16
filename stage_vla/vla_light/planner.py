"""planner.py —— 指令 → 基础动作计划（把指令分解成基础动作序列）。

两种计划来源：
1. **语义计划**（确定、正确）：复用 ``stages.semantic.SemanticSeparator``，把指令
   解析成语义阶段序列 —— 阶段即基础动作（approach/grasp/lift/move/stack）。
2. **可学习计划头**（``PlanDecoder``）：指令嵌入 → 基础动作序列 logits
   ``[B, plan_len, n_primitives]``，可训练，用于学会把任意指令分解成基础动作。

模型输出计划 logits；语义计划作为**交叉验证 / 训练监督**。
"""

from __future__ import annotations

import torch
from torch import nn

from .primitives import PRIMITIVES, PRIMITIVE_NAMES


def semantic_plan(instruction: str, stages: list[str] | None = None) -> list[str]:
    """用语义分离器把指令解析成基础动作序列（确定性）。

    Args:
        instruction: 语言指令
        stages: 允许的基础动作名（缺省用 PRIMITIVES 全量）
    """
    from ..stages.semantic import SemanticSeparator

    allowed = stages or PRIMITIVE_NAMES
    plan = SemanticSeparator().parse(instruction).stages
    return [s for s in plan if s in allowed]


# 计划输出的 STOP 标记索引（= 基础动作数；短计划尾部填充）
STOP_IDX = len(PRIMITIVES)
N_PRIMITIVES = len(PRIMITIVES)


class PlanDecoder(nn.Module):
    """可学习计划头：指令嵌入 → 基础动作序列 logits（含 STOP 标记）。

    输出 ``[B, plan_len, n_primitives+1]``，最后一个位置是 STOP：短计划的尾部
    学输出 STOP，解码时读到 STOP 即止（避免尾部垃圾）。配合语义计划监督训练后，
    模型即可把任意新指令分解成基础动作序列。
    """

    def __init__(self, instr_dim: int, n_primitives: int = N_PRIMITIVES, plan_len: int = 5, hidden: int = 256):
        super().__init__()
        self.n_primitives = n_primitives
        self.plan_len = plan_len
        self.vocab = n_primitives + 1           # 基础动作 + STOP
        self.net = nn.Sequential(
            nn.Linear(instr_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.vocab * plan_len),
        )

    def forward(self, instr_emb: torch.Tensor) -> torch.Tensor:
        """instr_emb: [B, instr_dim] → logits [B, plan_len, n_primitives+1]。"""
        B = instr_emb.shape[0]
        return self.net(instr_emb).view(B, self.plan_len, self.vocab)

    def decode(self, instr_emb: torch.Tensor) -> list[list[str]]:
        """argmax 解码成基础动作名序列（读到 STOP 即止）。"""
        logits = self.forward(instr_emb)
        ids = logits.argmax(dim=-1)             # [B, plan_len]
        plans = []
        for row in ids.tolist():
            plan = []
            for i in row:
                if i == self.n_primitives:
                    break                       # 遇 STOP 停止
                plan.append(PRIMITIVES[i].name)
            plans.append(plan)
        return plans
