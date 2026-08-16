"""instruction_tokenizer.py —— 指令分词器（轻量文本编码器，替代 OpenVLA 的 Llama-2 7B）。

把语言指令**即时**编码成固定维指令嵌入，替代原 7B 因果 LM 的角色（理解指令）与
自回归动作生成（改为动作回归头）：

    指令文本 → T5 tokenizer → token IDs → 词嵌入 → 小型 transformer encoder
             → 均值池化 → 指令嵌入 [B, D]

体积：词嵌入 32100×512 ≈ 16M + 4 层 transformer 编码器 ≈ 12M ≈ **28M 参数**
（对比 Llama-2 7B ≈ 6.7B，缩小约 240×）。

tokenizer 复用本机 T5（``paths.t5_model``，只读引用，不下载）。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer

from ..core.config import Settings


class InstructionTokenizer(nn.Module):
    """轻量指令编码器：文本 → 指令嵌入（可训练，供微调/蒸馏）。"""

    def __init__(
        self,
        tokenizer_path: str,
        hidden: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        pool: str = "mean",
        max_len: int = 64,
    ):
        super().__init__()
        self.hidden = hidden
        self.pool = pool
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = self.tokenizer.vocab_size

        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=self.tokenizer.pad_token_id or 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> "InstructionTokenizer":
        return cls(str(settings.require_path("t5_model")), **kwargs)

    # ------------------------------------------------------------------
    def _tokenize(self, instructions: list[str]) -> dict:
        enc = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def forward(self, instructions: list[str] | str) -> torch.Tensor:
        """指令文本 → 指令嵌入 ``[B, hidden]``。

        - 不同指令 → 不同嵌入（tokenizer 即时编码，可处理任意新指令）
        - 池化：mean = 有效 token 嵌入的均值
        """
        if isinstance(instructions, str):
            instructions = [instructions]
        tok = self._tokenize(instructions)
        ids = tok["input_ids"].to(next(self.parameters()).device)
        mask = tok["attention_mask"].to(next(self.parameters()).device)

        x = self.embed(ids)                                   # [B, L, D]
        x = x + self.pos_embed[:, : x.shape[1], :]            # 位置嵌入

        pad_mask = mask == 0                                  # src_key_padding_mask: True=忽略
        x = self.encoder(x, src_key_padding_mask=pad_mask)    # [B, L, D]

        if self.pool == "mean":
            mask_f = mask.unsqueeze(-1).float()
            return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)   # [B, D]
        return x[:, -1, :]                                    # last-token 池化

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
