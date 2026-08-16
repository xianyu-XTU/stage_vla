"""vision_policy.py —— 视觉引导的基础动作策略（VisionPrimitivePolicy）。

三层架构，对应"拆基础动作 → 指令分解 → 视觉输出动作"：

1. **基础动作库**（``primitives.py``）：approach/grasp/lift/move/stack，带参数维度。
2. **指令 → 基础动作计划**（``planner.py``）：指令编码 → 基础动作序列（PlanDecoder，
   语义计划交叉验证）。
3. **视觉 → 下一个基础动作**：当前视觉特征 + 指令嵌入 + 计划当前步上下文
   → 下一个基础动作（离散分类）+ 参数（连续回归，按动作 mask）。

即模型根据**看到的视觉**决定当前应执行哪个基础动作及其参数。
"""

from __future__ import annotations

import torch
from torch import nn

from ..core.config import Settings
from ..rl.vla_policy import VisionFeatureExtractor
from .instruction_tokenizer import InstructionTokenizer
from .planner import PlanDecoder
from .primitives import MAX_PARAMS_DIM, PRIMITIVES, PRIMITIVE_NAMES, params_mask


class VisionPrimitivePolicy(nn.Module):
    """视觉引导的基础动作策略：指令分解成计划，视觉决定当前动作。"""

    def __init__(
        self,
        settings: Settings,
        *,
        vision_device: str = "cuda",
        tokenizer_hidden: int = 512,
        plan_len: int = 5,
        hidden: int = 256,
    ):
        super().__init__()
        self.n_primitives = len(PRIMITIVES)
        self.plan_len = plan_len
        self.max_params_dim = MAX_PARAMS_DIM

        # 冻结视觉塔（纯视觉特征）
        self.vision = VisionFeatureExtractor(settings, device=vision_device, include_lang=False)
        # 指令编码 → 指令嵌入
        self.instruction_tokenizer = InstructionTokenizer.from_settings(settings, hidden=tokenizer_hidden)
        # 指令嵌入 → 基础动作计划
        self.plan_decoder = PlanDecoder(tokenizer_hidden, self.n_primitives, plan_len)
        # 视觉+指令+步骤上下文 → 下一个基础动作 + 参数
        fused_in = self.vision.features_dim + tokenizer_hidden + plan_len
        self.primitive_head = nn.Linear(fused_in, self.n_primitives)
        self.param_head = nn.Linear(fused_in, self.max_params_dim)

        self.instruction_tokenizer.to(vision_device)
        self.plan_decoder.to(vision_device)
        self.primitive_head.to(vision_device)
        self.param_head.to(vision_device)

    # ------------------------------------------------------------------
    def _fuse(self, image, instr_emb: torch.Tensor, step_idx) -> torch.Tensor:
        visual = self.vision(image)                                   # [B, 4096]
        step_ctx = self._step_onehot(step_idx, visual.shape[0], visual.device)  # [B, plan_len]
        return torch.cat([visual, instr_emb, step_ctx], dim=1)        # [B, fused_in]

    @staticmethod
    def _step_onehot(step_idx, B: int, device) -> torch.Tensor:
        idx = torch.full((B,), int(step_idx), device=device, dtype=torch.long).clamp(0, 100)
        return nn.functional.one_hot(idx, num_classes=100).float()[:, :5]

    # ------------------------------------------------------------------
    def forward(self, image, instruction, step_idx: int) -> dict:
        """一次推理：给定图像 + 指令 + 计划当前步 → 下一个基础动作 + 参数。

        Returns:
            {"plan_logits": [B, plan_len, n_primitives],  指令分解出的计划
             "next_primitive": [B, n_primitives],         视觉决定的当前动作
             "params": [B, max_params_dim]}               动作参数（按动作 mask）
        """
        instr_emb = self.instruction_tokenizer(instruction)          # [B, hidden]
        instr_emb = instr_emb.to(next(self.primitive_head.parameters()).device)
        plan_logits = self.plan_decoder(instr_emb)                   # [B, plan_len, n_primitives]

        fused = self._fuse(image, instr_emb, step_idx)
        next_logits = self.primitive_head(fused)                     # [B, n_primitives]
        params = self.param_head(fused)                              # [B, max_params_dim]
        return {"plan_logits": plan_logits, "next_primitive": next_logits, "params": params}

    def predict_step(self, image, instruction, step_idx: int) -> tuple[str, list[float], list[str]]:
        """推理别名：返回 (下一个基础动作名, 参数(按动作 mask), 计划动作名序列)。"""
        out = self.forward(image, instruction, step_idx)
        B = out["next_primitive"].shape[0]
        prim_ids = out["next_primitive"].argmax(dim=-1)
        names = [PRIMITIVES[int(i)].name for i in prim_ids]
        params = []
        for b in range(B):
            pid = int(prim_ids[b])
            mask = torch.tensor(params_mask(PRIMITIVES[pid].name), dtype=out["params"].dtype,
                                device=out["params"].device)
            params.append((out["params"][b] * mask).tolist())
        plan = [[PRIMITIVES[i].name for i in row] for row in out["plan_logits"].argmax(dim=-1).tolist()]
        return names, params, plan

    def param_report(self) -> dict:
        return {
            "vision (frozen)": sum(p.numel() for p in self.vision.parameters()),
            "instruction_tokenizer": self.instruction_tokenizer.param_count(),
            "plan_decoder": sum(p.numel() for p in self.plan_decoder.parameters()),
            "primitive/param heads": sum(p.numel() for p in self.primitive_head.parameters())
                                    + sum(p.numel() for p in self.param_head.parameters()),
        }

    def trainable_parameters(self):
        params = (list(self.instruction_tokenizer.parameters())
                  + list(self.plan_decoder.parameters())
                  + list(self.primitive_head.parameters())
                  + list(self.param_head.parameters()))
        for p in params:
            p.requires_grad_(True)
        return params
