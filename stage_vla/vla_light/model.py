"""model.py —— 去 LLM 的轻量 VLA 模型（OpenVLALightForAction）。

替换 OpenVLA 的 Llama-2 7B 因果 LM 为**指令分词器**（小型 transformer 编码器）：

    图像 → VisionFeatureExtractor(冻结, include_lang=False) → 视觉特征 [B, 4096]
    指令 → InstructionTokenizer(可训练)                     → 指令嵌入 [B, hidden]
    拼接 → action_head MLP → 7 维动作（IK-Rel：6 末端增量 + 1 夹爪，回归）

关键差异 vs 原 OpenVLA：
- 无 7B LLM（指令用轻量编码器即时编码，可处理任意新指令）
- 动作**回归**而非自回归文本生成（无需 generate/past_key_values/256-bin 离散）

体积：视觉塔 0.77B（冻结）+ 指令分词器 ~28M + 动作头 ~1.2M ≈ **0.8B**（对比 7B，-88%）。
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.config import Settings
from ..rl.vla_policy import VisionFeatureExtractor
from .instruction_tokenizer import InstructionTokenizer


class OpenVLALightForAction(nn.Module):
    """轻量 VLA：视觉塔(冻结) + 指令分词器(可训练) + 动作回归头。"""

    def __init__(
        self,
        settings: Settings,
        *,
        vision_device: str = "cuda",
        tokenizer_hidden: int = 512,
        action_dim: int = 7,
        hidden: int = 256,
    ):
        super().__init__()
        # 冻结视觉塔（纯视觉，不含预编码指令）
        self.vision = VisionFeatureExtractor(settings, device=vision_device, include_lang=False)
        # 可训练指令分词器（文本 → 指令嵌入）
        self.instruction_tokenizer = InstructionTokenizer.from_settings(
            settings, hidden=tokenizer_hidden,
        )
        self.action_dim = action_dim

        head_in = self.vision.features_dim + tokenizer_hidden   # 4096 + hidden
        self.action_head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # 分词器 + 动作头随视觉塔放 GPU（tokenizer 文本处理仍 CPU，embedding/encoder 上 GPU）
        self.instruction_tokenizer.to(vision_device)
        self.action_head.to(vision_device)

    # ------------------------------------------------------------------
    def forward(self, image, instructions: list[str] | str) -> torch.Tensor:
        """图像 + 指令 → 7 维动作 ``[B, action_dim]``。

        image: ``[H,W,3]`` uint8 或 ``[B,H,W,3]``
        instructions: 指令文本（列表或单条）
        """
        visual = self.vision(image)                              # [1, 4096] 或 [B, 4096]（GPU，冻结）
        lang = self.instruction_tokenizer(instructions)          # [B, hidden]（CPU）
        lang = lang.to(visual.device)

        # 单图 × 多指令 → 广播视觉特征到指令 batch
        if visual.shape[0] == 1 and lang.shape[0] > 1:
            visual = visual.expand(lang.shape[0], -1)

        x = torch.cat([visual, lang], dim=1)                     # [B, 4096+hidden]
        return self.action_head(x)                               # [B, 7]

    def predict_action(self, image, instructions: list[str] | str) -> torch.Tensor:
        """推理别名（对齐 OpenVLA 的 predict_action 语义）。"""
        return self.forward(image, instructions)

    def param_report(self) -> dict:
        """参数/体积报告（对比 OpenVLA 7B）。"""
        report = {
            "vision (frozen)": sum(p.numel() for p in self.vision.parameters()),
            "instruction_tokenizer": self.instruction_tokenizer.param_count(),
            "action_head": sum(p.numel() for p in self.action_head.parameters()),
        }
        report["total"] = sum(report.values())
        return report

    def trainable_parameters(self):
        """只训练指令分词器 + 动作头（视觉塔冻结）。"""
        params = list(self.instruction_tokenizer.parameters()) + list(self.action_head.parameters())
        for p in params:
            p.requires_grad_(True)
        return params
