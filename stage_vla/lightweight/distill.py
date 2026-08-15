"""distill.py —— 知识蒸馏（模块③ 核心预留，M3 实现）。

申请书承诺：以优化后的大模型为教师模型，训练小型轻量化模型作为学生模型，把教师
知识迁移到学生，平衡轻量化与性能损失。

方案（M3 落地）：
- teacher = OpenVLA-7B（或 LoRA 微调后的版本）
- student = vision_only 后端 / LoRA-INT8
- loss = logits 匹配（KL，temperature=T）+ 可选 feature 匹配 + 任务 loss（alpha 平衡）

M0 只定义接口与配置，``train_student`` 抛指引性错误。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.config import Settings


@dataclass
class DistillConfig:
    """蒸馏超参（与 config/lightweight.distill 对齐）。"""

    teacher_model: str = "openvla-7b"
    student_backend: str = "vision_only"
    temperature: float = 3.0
    alpha: float = 0.5          # 蒸馏 loss 与任务 loss 的平衡系数
    loss_weights: dict = field(default_factory=lambda: {"logits": 1.0, "feature": 0.0})

    @classmethod
    def from_settings(cls, settings: Settings) -> "DistillConfig":
        cfg = settings.lightweight.get("distill", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class Distiller(ABC):
    """教师-学生蒸馏器基类。"""

    @abstractmethod
    def train_student(self, demo_loader, num_epochs: int, log_dir: str) -> None:
        """用演示数据 + 教师输出蒸馏学生模型。"""
        raise NotImplementedError

    @abstractmethod
    def evaluate_student(self) -> dict:
        """返回学生模型在验证集上的指标（动作预测一致性 / 成功率）。"""
        raise NotImplementedError


def distill(
    settings: Settings,
    demo_dir: str,
    num_epochs: int = 10,
    log_dir: str = "outputs/checkpoints/distill",
) -> object:
    """知识蒸馏入口（M3 实现）。

    Raises:
        NotImplementedError: M3 里程碑实现，当前仅占位。
    """
    raise NotImplementedError(
        "知识蒸馏为 M3 里程碑（teacher=OpenVLA-7B → student=vision_only/LoRA-INT8）。"
        "参见 docs/module3_lightweight.md。"
    )
