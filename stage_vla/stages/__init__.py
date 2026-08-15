"""模块① 长程任务阶段感知机制。

- ``semantic``      语义分离（指令 → 子目标 → 阶段序列）
- ``calculator``    阶段计算器（纯函数：信号 → 阶段索引/进度，单一实现）
- ``detector``      几何阶段检测器（原语 + 委托 calculator）
- ``unsupervised``  无监督阶段分离器（M1 研究项，占位）
- ``rewards``       稠密奖励纯张量函数（势能塑形 + 阶段完成奖）
- ``rewards_isaac`` Isaac Lab RewardTerm 适配层（需 Isaac 环境）
"""

from .detector import DEFAULT_STAGES, DEFAULT_THRESHOLDS, StageDetector
from .semantic import (
    ACTION_KEYWORDS,
    COLOR_TO_ENTITY,
    SemanticPlan,
    SemanticSeparator,
    SemanticSubGoal,
)
from .unsupervised import (
    assign_stages,
    evaluate_consistency,
    extract_features,
    fit_unsupervised_centers,
)

__all__ = [
    "ACTION_KEYWORDS",
    "COLOR_TO_ENTITY",
    "DEFAULT_STAGES",
    "DEFAULT_THRESHOLDS",
    "SemanticPlan",
    "SemanticSeparator",
    "SemanticSubGoal",
    "StageDetector",
    "assign_stages",
    "evaluate_consistency",
    "extract_features",
    "fit_unsupervised_centers",
]
