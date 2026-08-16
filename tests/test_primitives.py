"""基础动作库 + 指令→计划 测试（纯 torch，无需视觉模型）。"""

from __future__ import annotations

import torch

from stage_vla.vla_light.planner import STOP_IDX, PlanDecoder, semantic_plan
from stage_vla.vla_light.primitives import MAX_PARAMS_DIM, PRIMITIVES, params_mask, primitive_by_name


def test_primitive_library():
    """基础动作库：名称唯一、参数维度合理、mask 正确。"""
    names = [p.name for p in PRIMITIVES]
    assert len(names) == len(set(names)), "动作名应唯一"
    for p in PRIMITIVES:
        assert p.params_dim <= MAX_PARAMS_DIM
        assert p.controller in ("ik_move", "close_gripper", "open_gripper")
    # 参数 mask：有参数的维为 1，无参数的为 0
    mask_move = params_mask("move")
    assert sum(mask_move) == 3
    mask_grasp = params_mask("grasp")
    assert sum(mask_grasp) == 0


def test_semantic_plan_full():
    """完整指令 → 五阶段基础动作序列。"""
    plan = semantic_plan("pick up the red cube and place it on the blue cube")
    assert plan == ["approach", "grasp", "lift", "move", "stack"]


def test_semantic_plan_pick_only():
    """只抓不放 → 只有前三个阶段。"""
    plan = semantic_plan("just grab the green cube")
    assert plan == ["approach", "grasp", "lift"]


def test_plan_decoder_shape_and_decode():
    """计划头：输出含 STOP，argmax 解码在 STOP 处停止。"""
    decoder = PlanDecoder(instr_dim=128, n_primitives=5, plan_len=5)
    emb = torch.randn(2, 128)
    logits = decoder(emb)
    assert tuple(logits.shape) == (2, 5, 6)   # 5 动作 + 1 STOP
    plans = decoder.decode(emb)
    assert len(plans) == 2
    assert all(0 < len(p) <= 5 for p in plans)   # 随机初始化可能提前 STOP
    assert all(name in {p.name for p in PRIMITIVES} for name in plans[0])


def test_plan_decoder_trains_to_match_semantic():
    """过拟合一个小批次：让计划头学会 approach→grasp→lift→STOP→STOP。"""
    torch.manual_seed(0)
    decoder = PlanDecoder(instr_dim=16, n_primitives=5, plan_len=5)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-2)
    # 目标：approach→grasp→lift→STOP→STOP（索引 0,1,2,5,5）
    emb = torch.randn(8, 16)
    target = torch.tensor([0, 1, 2, STOP_IDX, STOP_IDX]).repeat(8, 1)
    for _ in range(300):
        logits = decoder(emb)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 6), target.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    decoded = decoder.decode(emb)
    assert all(d == ["approach", "grasp", "lift"] for d in decoded), f"未学会：{decoded[0]}"
