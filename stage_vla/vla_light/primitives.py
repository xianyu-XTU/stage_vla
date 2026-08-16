"""primitives.py —— 机械臂基础动作库（action primitives）。

把机械臂的长程操作拆分为**基础动作**（primitive），每个基础动作是带参数的高级技能，
由低级控制器（IK/夹爪）执行。与阶段框架（approach/grasp/lift/move/stack）对齐。

每个基础动作：
- ``name``：动作名（同时也是语义阶段名）
- ``params_dim``：连续参数维度（如 approach/move 的目标位置 3 维、lift 的高度增量 1 维）
- ``controller``：低级执行方式（目前标注，供仿真接线）

模型输出 = 下一个基础动作（离散分类）+ 参数（连续回归，按动作 mask）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    params_dim: int
    controller: str
    description: str


# 基础动作库（对齐阶段框架，IK-Rel 7 维控制空间）
PRIMITIVES: list[Primitive] = [
    Primitive("approach", params_dim=3, controller="ik_move", description="末端移动到目标位置"),
    Primitive("grasp", params_dim=0, controller="close_gripper", description="闭合夹爪抓取"),
    Primitive("lift", params_dim=1, controller="ik_move", description="抬升被夹物体到指定高度"),
    Primitive("move", params_dim=3, controller="ik_move", description="移动到目标/底座位置"),
    Primitive("stack", params_dim=0, controller="open_gripper", description="在目标位置释放/堆叠"),
]

# 名称 → Primitive 索引
PRIMITIVE_INDEX = {p.name: i for i, p in enumerate(PRIMITIVES)}
MAX_PARAMS_DIM = max(p.params_dim for p in PRIMITIVES)

# 名称列表（供计划/分类用）
PRIMITIVE_NAMES = [p.name for p in PRIMITIVES]


def primitive_by_name(name: str) -> Primitive:
    return PRIMITIVES[PRIMITIVE_INDEX[name]]


def params_mask(name: str, max_dim: int = MAX_PARAMS_DIM) -> list[float]:
    """某基础动作的参数 mask：有效参数维 1，其余 0（用于回归输出裁剪）。"""
    dim = primitive_by_name(name).params_dim
    return [1.0 if i < dim else 0.0 for i in range(max_dim)]
