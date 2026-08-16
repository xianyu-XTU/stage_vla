"""稠密奖励测试：势能塑形正负 / 跨级完成奖 / 首帧 mask。"""

from __future__ import annotations

import torch

from stage_vla.stages.rewards import (
    first_frame_mask,
    potential_shaping,
    stage_completion_reward,
    stage_completion_reward_first_time,
)


def test_potential_shaping_sign():
    cur = torch.tensor([[0.8, 0.0, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0]])
    prev = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0]])
    r = potential_shaping(cur, prev)
    assert r[0].item() > 0  # 进步 → 正
    assert r[1].item() < 0  # 退步 → 负


def test_potential_shaping_gamma():
    cur = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    r = potential_shaping(cur, prev, gamma=0.99)
    assert abs(r.item() - 0.99) < 1e-6


def test_stage_completion_cross_level(settings):
    stage = torch.tensor([3, 2, 1, 4])
    prev_stage = torch.tensor([1, 0, 1, 3])
    w = settings.reward_weights
    bonus = stage_completion_reward(stage, prev_stage, w, settings.stages)
    assert abs(bonus[0].item() - (w["lift"] + w["move"])) < 1e-5
    assert abs(bonus[1].item() - (w["grasp"] + w["lift"])) < 1e-5
    assert bonus[2].item() == 0.0
    assert abs(bonus[3].item() - w["stack"]) < 1e-5


def test_first_frame_mask():
    episode_length_buf = torch.tensor([1, 5, 1, 3])
    mask = first_frame_mask(episode_length_buf)
    assert mask.tolist() == [True, False, True, False]


def test_stage_completion_first_time_only(settings):
    """防奖励黑客：同一阶段反复跨入只奖励首次。"""
    stages = settings.stages
    w = settings.reward_weights
    N = 1
    done0 = torch.zeros(N, len(stages), dtype=torch.bool)
    # env0: 0→2（首次进入 1,2）
    bonus, done1 = stage_completion_reward_first_time(
        torch.tensor([2]), torch.tensor([0]), done0, w, stages)
    assert abs(bonus[0].item() - (w["grasp"] + w["lift"])) < 1e-5
    # 再次跨入（1→2），不应再发（已标记 done）
    bonus2, done2 = stage_completion_reward_first_time(
        torch.tensor([2]), torch.tensor([1]), done1, w, stages)
    assert bonus2[0].item() == 0.0, "已奖励过的阶段不应重复给奖"
    # 但新阶段 3（move）应发
    bonus3, done3 = stage_completion_reward_first_time(
        torch.tensor([3]), torch.tensor([1]), done1, w, stages)
    assert abs(bonus3[0].item() - w["move"]) < 1e-5
    assert bool(done3[0, 3]) is True
