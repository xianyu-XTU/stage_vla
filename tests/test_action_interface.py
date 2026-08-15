"""动作输出接口测试：IKRel 夹爪二进制化、JointPos 128 维往返。"""

from __future__ import annotations

import torch

from stage_vla.rl.action_interface import (
    IKRelActionInterface,
    JointPosActionInterface,
    decode_action_128,
    encode_state_128,
)

# 合成 rdt_normalization（关节域接近、夹爪域不同，贴合真实结构）
_NORM = {
    "state_min": [-0.75, -0.08, -0.50, -2.66, -0.57, 1.83, -2.24, 0.0],
    "state_max": [0.76, 1.50, 0.47, -0.39, 0.55, 3.29, 2.57, 0.04],
    "action_min": [-0.75, -0.08, -0.50, -2.66, -0.57, 1.83, -2.24, -1.0],
    "action_max": [0.76, 1.50, 0.47, -0.39, 0.55, 3.29, 2.57, 1.0],
}


def _interface() -> JointPosActionInterface:
    n = _NORM
    return JointPosActionInterface(n["state_min"], n["state_max"], n["action_min"], n["action_max"])


def test_ikrel_gripper_binary():
    """IKRel：6 维末端增量直通，夹爪符号 → 二进制 ±1。"""
    iface = IKRelActionInterface()
    vla = torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.8],   # 夹爪 0.8 → 开 +1
                        [-0.1, 0.2, -0.05, 0.0, 0.0, 0.0, -0.3]])  # 夹爪 -0.3 → 夹 -1
    env = iface.to_env_action(vla)
    assert env.shape == (2, 7)
    torch.testing.assert_close(env[0, :6], vla[0, :6])   # 6 维直通
    assert env[0, 6].item() == 1.0                        # 夹爪开
    assert env[1, 6].item() == -1.0                       # 夹爪夹


def test_ikrel_from_env_state_is_none():
    assert IKRelActionInterface().from_env_state(None) is None


def test_jointpos_encode_state_128():
    """8 维关节状态 → 128 维归一化域。"""
    state = torch.tensor([[0.0, 0.7, 0.0, -1.5, 0.0, 2.5, 0.0, 0.02]])
    enc = encode_state_128(state, torch.tensor(_NORM["state_min"]), torch.tensor(_NORM["state_max"]))
    assert enc.shape == (1, 128)
    # 中间值应在 [-1,1]；0.02 在 [0,0.04] → 归一化 ≈ 0
    assert bool((enc[0, :8] >= -1).all()) and bool((enc[0, :8] <= 1).all())
    assert bool((enc[0, 8:] == 0).all())  # 其余 120 维为 0


def test_jointpos_roundtrip_joint_dims():
    """关节维度（0-6，state/action 边界相同）往返应恢复原值。"""
    iface = _interface()
    state = torch.tensor([[0.1, 0.5, -0.2, -1.0, 0.3, 2.0, -0.5, 0.02]])
    enc = iface.from_env_state(state)          # [N,128]
    dec = iface.to_env_action(enc)             # [N,8]
    # 关节 0-6：边界一致 → 应恢复；夹爪 7：边界不同 → 不要求
    torch.testing.assert_close(dec[0, :7], state[0, :7], atol=1e-4, rtol=1e-4)
    assert bool((dec[0, 7] >= -1).all())       # 夹爪在 action 域


def test_jointpos_roundtrip_128_const():
    """encode→decode 用同一套边界应近似恒等。"""
    state = torch.tensor([0.2, -0.1, 0.05, -1.2, 0.4, 2.3, -0.6, 0.0])
    s_min, s_max = torch.tensor(_NORM["state_min"]), torch.tensor(_NORM["state_max"])
    enc = encode_state_128(state, s_min, s_max)
    dec = decode_action_128(enc, s_min, s_max)   # 用同一 state 边界
    torch.testing.assert_close(dec, state, atol=1e-4, rtol=1e-4)
