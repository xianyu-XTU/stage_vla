"""vla_light —— 去 LLM 的轻量 VLA（指令分词器替代 Llama-2 7B 语言模型）。

OpenVLA 用 Llama-2 7B 因果 LM 既理解指令又自回归生成动作 token（体积大头 ~6.7B）。
本包把它替换为**指令分词器**（T5 tokenizer + 小型 transformer encoder，~28M 参数），
架构变为：

    图像 → 视觉塔(冻结) → 投影器 → 视觉特征
    指令 → 指令分词器 → 指令嵌入
    拼接 → 动作回归头 → 7 维动作（无自回归生成）

体积：OpenVLA ~7B → 本实现 ~0.8B（视觉塔 0.77B + 指令分词器 28M + 动作头 1.2M）。
"""

from .instruction_tokenizer import InstructionTokenizer
from .model import OpenVLALightForAction
from .planner import PlanDecoder, semantic_plan
from .primitives import PRIMITIVES, PRIMITIVE_NAMES
from .vision_policy import VisionPrimitivePolicy

__all__ = [
    "InstructionTokenizer",
    "OpenVLALightForAction",
    "PlanDecoder",
    "PRIMITIVES",
    "PRIMITIVE_NAMES",
    "VisionPrimitivePolicy",
    "semantic_plan",
]
