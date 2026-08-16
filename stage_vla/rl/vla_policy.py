"""vla_policy.py —— VLA-as-policy（模块② 核心，M2 实现）。

申请书承诺"以 OpenVLA 为基础 + PPO 构建 StARe-PPO"，即把视觉语言策略当作 PPO 的
**actor**。rsl_rl 的 OnPolicyRunner 期望 MLP 风格 actor_critic，与 VLA 策略形态不匹配，
因此 M2 用**自研 PPO 循环**（见 ``ppo_loop.py``），策略实现本接口：

    VLAAsPolicy(nn.Module)
      ├── act(obs, stage_feedback) -> (action, log_prob, value)     # 采样动作 + 价值
      └── evaluate_actions(obs, action, stage_feedback)             # 评估（log_prob/entropy/value）
            -> (log_prob, entropy, value)

``VisionOnlyPPOPolicy`` 结构：
- ``feature_extractor``：把观测（图像/指令/状态）编码成特征 ``[B, F]``。M2a 用
  ``DenseFeatureExtractor``（obs 即特征，可纯 torch 单测）；M2b 换成视觉塔提取器。
- ``actor_head`` / ``critic_head``：小 MLP 输出动作均值 / 价值；``log_std`` 可学习。
- **8GB 纪律**：视觉塔冻结（前向 detach/bf16），只训练动作头与价值头（2.1M 参数量级）。
- ``stage_feedback`` 作为额外条件输入拼接进特征（阶段感知融合）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


# ============================================================================
# 特征提取器（可插拔：M2a 稠密特征 / M2b 视觉塔）
# ============================================================================
class FeatureExtractor(nn.Module):
    """把观测编码成策略特征 ``[B, F]``。"""

    @abstractmethod
    def forward(self, obs) -> torch.Tensor:
        raise NotImplementedError


class DenseFeatureExtractor(FeatureExtractor):
    """M2a 测试/稠密模式：观测本身即特征（无视觉塔）。"""

    def __init__(self, features_dim: int):
        super().__init__()
        self.features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x


class VisionFeatureExtractor(FeatureExtractor):
    """冻结视觉塔（DINOv2-L + SigLIP SO400M → 投影器）+ 指令嵌入 → ``[B, 2*llm_dim]``。

    **8GB 纪律**：视觉塔冻结（``requires_grad=False``，前向 bf16），显存 ~1.5GB；
    只训练动作/价值头。加载本机剥离的视觉模型（``paths.vision_only_dir``），
    指令嵌入来自 ``paths.lang_embed``（T5 预编码，固定指令）。
    """

    def __init__(self, settings, device: str = "cuda", include_lang: bool = True):
        super().__init__()
        self.device = device
        self.include_lang = include_lang
        self._load(settings)

    # ------------------------------------------------------------------
    def _load(self, settings) -> None:
        from ..policies.prismatic import ensure_prismatic_importable  # 惰性，仅视觉模式

        openvla_model = settings.require_path("openvla_model")
        vision_dir = settings.require_path("vision_only_dir")
        lang_embed_path = settings.require_path("lang_embed")
        ensure_prismatic_importable(settings)

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import (
            PrismaticProjector,
            PrismaticVisionBackbone,
        )
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

        meta = torch.load(str(vision_dir / "meta.pt"), map_location="cpu")
        sd = torch.load(str(vision_dir / "vision_only.pt"), map_location="cpu")

        cfg = OpenVLAConfig.from_pretrained(str(openvla_model))
        self.vision_backbone = PrismaticVisionBackbone(
            cfg.use_fused_vision_backbone, cfg.image_sizes,
            cfg.timm_model_ids, cfg.timm_override_act_layers,
        )
        self.projector = PrismaticProjector(
            cfg.use_fused_vision_backbone, self.vision_backbone.embed_dim,
            cfg.text_config.hidden_size,
        )

        vb_sd = {k[len("vision_backbone."):]: v for k, v in sd.items() if k.startswith("vision_backbone.")}
        pj_sd = {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}
        self.vision_backbone.load_state_dict(vb_sd)
        self.projector.load_state_dict(pj_sd)
        # 冻结 + bf16
        self.vision_backbone = self.vision_backbone.to(self.device, dtype=torch.bfloat16).eval()
        self.projector = self.projector.to(self.device, dtype=torch.bfloat16).eval()
        for p in self.vision_backbone.parameters():
            p.requires_grad_(False)
        for p in self.projector.parameters():
            p.requires_grad_(False)

        self.image_processor = PrismaticImageProcessor.from_pretrained(str(openvla_model))

        self.llm_dim = int(meta["llm_dim"])

        # 指令嵌入（可选）：兼容 [L,4096] 与 [1,L,4096]，均值池化 → [1, llm_dim]
        if self.include_lang:
            lang = torch.load(str(lang_embed_path), map_location="cpu")
            emb = lang["embeddings"].to(self.device)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            else:
                emb = emb.mean(dim=0)
            self.lang_embed = emb.unsqueeze(0)

        # 纯视觉 = llm_dim；带预编码指令 = 2*llm_dim
        self.features_dim = (2 if self.include_lang else 1) * self.llm_dim
        print(f"[VisionFeatureExtractor] 就绪: features_dim={self.features_dim} "
              f"(include_lang={self.include_lang}), 显存 {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ------------------------------------------------------------------
    def forward(self, image) -> torch.Tensor:
        """图像 ``[H,W,3]`` uint8 或 ``[B,H,W,3]`` → ``[B, 2*llm_dim]`` 冻结特征。"""
        import numpy as np

        if torch.is_tensor(image):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[None]
        batch = []
        for img in arr:
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
            pixel = self.image_processor.apply_transform(img_t).to(self.device, dtype=torch.bfloat16)
            batch.append(pixel)
        pixel_values = torch.cat(batch, dim=0)  # [B,6,224,224]

        with torch.no_grad():
            feats = self.vision_backbone(pixel_values)       # [B, N, vis_dim]
            proj = self.projector(feats)                     # [B, N, llm_dim]
            visual = proj.mean(dim=1).float()                # [B, llm_dim]
            if self.include_lang:
                lang = self.lang_embed.float().expand(visual.shape[0], -1)  # [B, llm_dim]
                out = torch.cat([visual, lang], dim=1)       # [B, 2*llm_dim]
            else:
                out = visual                                # [B, llm_dim]（纯视觉，指令由外部编码）
        return out


# ============================================================================
# VLA 策略接口
# ============================================================================
class VLAAsPolicy(nn.Module, ABC):
    """可被 PPO 当作 actor 的 VLA 策略。"""

    #: 动作维度
    action_dim: int

    @abstractmethod
    def act(
        self,
        obs,
        stage_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作。

        Returns:
            (action [B, action_dim], log_prob [B], value [B])
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self,
        obs,
        action: torch.Tensor,
        stage_feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作。

        Returns:
            (log_prob [B], entropy [B], value [B])
        """
        raise NotImplementedError


# ============================================================================
# VisionOnlyPPOPolicy（M2 实现）
# ============================================================================
class _MLPHead(nn.Module):
    """小 MLP：特征 → 输出。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionOnlyPPOPolicy(VLAAsPolicy):
    """冻结视觉特征 + 可训练动作/价值头 的 PPO 策略。

    Parameters
    ----------
    features_dim : 特征维度（视觉塔输出 F，或稠密模式下的输入维）
    action_dim : 动作维度（IK-Rel 7）
    feature_extractor : 观测 → [B, F] 的提取器；None 时 obs 即特征（稠密模式）
    stage_feedback_dim : 阶段反馈条件输入维度（0 = 不用）
    init_log_std : 初始对数标准差
    """

    action_dim = 7

    def __init__(
        self,
        features_dim: int,
        action_dim: int = 7,
        *,
        feature_extractor: FeatureExtractor | None = None,
        stage_feedback_dim: int = 0,
        init_log_std: float = -1.0,
        hidden: int = 256,
        device: str | None = None,
    ):
        super().__init__()
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.feature_extractor = feature_extractor
        self.stage_feedback_dim = stage_feedback_dim
        head_in = features_dim + stage_feedback_dim
        self.actor_head = _MLPHead(head_in, action_dim, hidden)
        self.critic_head = _MLPHead(head_in, 1, hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        # 头常驻 CPU（见 _head_input 注释）；feature_extractor 自带设备（GPU 视觉塔）
        self._device_hint = device  # 保留参数兼容；实际头在 CPU

    # ------------------------------------------------------------------
    # 特征 + 阶段反馈 → 头输入
    # ------------------------------------------------------------------
    def _head_input(self, obs, stage_feedback: torch.Tensor | None) -> torch.Tensor:
        x = self.feature_extractor(obs) if self.feature_extractor is not None else torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if stage_feedback is not None:
            fb = torch.as_tensor(stage_feedback, dtype=torch.float32).to(x.device)
            x = torch.cat([x, fb], dim=-1)
        # 可训练头**常驻 CPU**（M2 实测：GPU backward 与 Isaac RTX 渲染/Warp 同卡触发
        # device assert；头仅 4.2M 参数，CPU 足够快）。特征由冻结视觉塔在 GPU 计算后 detach。
        return x.detach().cpu()

    def _distribution(self, x: torch.Tensor):
        mean = self.actor_head(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------
    def act(self, obs, stage_feedback=None):
        x = self._head_input(obs, stage_feedback)
        dist = self._distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, action, stage_feedback=None):
        x = self._head_input(obs, stage_feedback)
        return self.evaluate_x(x, action)

    # ------------------------------------------------------------------
    # 预提取特征模式（PPO 循环用：一次视觉前向，多次头更新，省显存/耗时）
    # ------------------------------------------------------------------
    def features(self, obs, stage_feedback=None) -> torch.Tensor:
        """提取头输入 ``x = 特征 + 阶段反馈``，供预存储与多次更新复用。"""
        return self._head_input(obs, stage_feedback)

    def sample_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对预提取的 ``x`` 采样动作（action, log_prob, value）。"""
        dist = self._distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return action, log_prob, value

    def evaluate_x(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对预提取的 ``x`` 评估动作（log_prob, entropy, value）。"""
        dist = self._distribution(x)
        a = torch.as_tensor(action, dtype=torch.float32).to(x.device)
        log_prob = dist.log_prob(a).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(x).squeeze(-1)
        return log_prob, entropy, value

    def value_x(self, x: torch.Tensor) -> torch.Tensor:
        """对预提取的 ``x`` 计算价值。"""
        return self.critic_head(x).squeeze(-1)

    def trainable_parameters(self):
        """只训练动作/价值头与 log_std（冻结特征提取器）。"""
        params = list(self.actor_head.parameters()) + list(self.critic_head.parameters()) + [self.log_std]
        for p in params:
            p.requires_grad_(True)
        return params
