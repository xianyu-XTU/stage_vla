"""rewards.py —— 阶段感知稠密奖励（纯张量函数层）。

将传统稀疏奖励转化为稠密监督，解决信用分配难题。两类奖励：
  1. 阶段势能塑形（每步稠密信号）：``r = γ·φ(s_t) − φ(s_{t−1})``，φ = 阶段进度和
  2. 阶段完成奖励：跨入新阶段时一次性给该阶段权重（approach=0 / grasp=2 / lift=1 /
     move=0.5 / stack=10）

本文件只含**纯张量函数**（可无 Isaac Sim 单测）；需要 ``env`` 的 Isaac Lab RewardTerm
函数与 RewardsCfg 工厂放在 :mod:`stage_vla.stages.rewards_isaac`。

回归点（吸收旧工程教训）：
- **首帧处理**：reward 在 reset 前计算且 ``episode_length_buf`` 已全量 +1，用
  ``episode_length_buf == 1`` 识别新回合首帧，把 prev 对齐 cur 并置零该帧塑形/完成奖，
  杜绝"刚 reset 首帧虚假势能差"。
- **dt 校正**：reward manager 会把每项 ``* dt``；Isaac 侧返回 ``值 / env.step_dt`` 抵消，
  使 config 权重即为每步名义值（见 ``rewards_isaac``）。
"""

from __future__ import annotations

import torch


def potential_shaping(
    cur_progress: torch.Tensor,
    prev_progress: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """势能塑形奖励：``γ·φ(s_t) − φ(s_{t−1})``。

    Args:
        cur_progress: [N, n_stages] 当前各阶段完成度
        prev_progress: [N, n_stages] 上一步各阶段完成度
        gamma: 势能折扣

    Returns:
        [N] 塑形奖励（无界，可乘权重）
    """
    return gamma * cur_progress.sum(dim=1) - prev_progress.sum(dim=1)


def stage_completion_reward(
    stage: torch.Tensor,
    prev_stage: torch.Tensor,
    weights: dict,
    stages: list[str],
) -> torch.Tensor:
    """阶段完成奖励：每当跨入新阶段，给予该阶段的完成权重。

    Args:
        stage: [N] 当前阶段索引
        prev_stage: [N] 上一步阶段索引
        weights: 阶段完成权重（键为阶段名，如 REWARD_WEIGHTS）
        stages: 阶段名列表（决定索引 → 权重映射）

    Returns:
        [N] 阶段完成奖励（只有刚进入新阶段时才 > 0，支持跨级累加）
    """
    bonus = torch.zeros_like(stage, dtype=torch.float)
    for level, name in enumerate(stages):
        entered_level = (level > prev_stage) & (level <= stage)
        bonus = bonus + entered_level.float() * weights[name]
    return bonus


def first_frame_mask(episode_length_buf: torch.Tensor) -> torch.Tensor:
    """判断哪些环境处于新回合首帧（reset 后第一帧）。

    依据：reward 在 reset 之前计算，而 step() 先把 ``episode_length_buf`` 全量 +1，
    所以刚 reset 的环境该帧 ``episode_length_buf == 1``。

    Returns:
        [N] bool
    """
    return episode_length_buf == 1


def stage_completion_reward_first_time(
    stage: torch.Tensor,
    prev_stage: torch.Tensor,
    done: torch.Tensor,
    weights: dict,
    stages: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """阶段完成奖励，**每个阶段每 episode 只奖励首次进入**（防奖励黑客）。

    背景（M2 长训练诊断）：原实现每次跨阶段都发完成奖，策略学会反复
    "抓了又掉、掉了再抓" 刷阶段完成奖（stage_transition 峰值 21 vs 正常 ~4），
    却从不真正完成堆叠（success=0）。本函数用 ``done`` 掩码只奖励首次。

    Args:
        stage: [N] 当前阶段索引
        prev_stage: [N] 上一步阶段索引
        done: [N, K] bool，本 episode 已奖励过的阶段
        weights: 阶段完成权重（键为阶段名）
        stages: 阶段名列表

    Returns:
        (bonus [N], new_done [N, K])
    """
    bonus = torch.zeros_like(stage, dtype=torch.float)
    new_done = done.clone()
    for level, name in enumerate(stages):
        entered = (level > prev_stage) & (level <= stage) & (~done[:, level])   # [N]
        bonus = bonus + entered.float() * weights[name]
        # 只更新该列（[N] 不能直接 | 到 [N,K]，否则会广播成整行全 True 的 bug）
        new_done[:, level] = new_done[:, level] | entered
    return bonus, new_done
