"""阶段成功状态库纯部分测试（add / sample / save / load，交接文档 11 节）。"""

from __future__ import annotations

import torch

from stage_vla.rl.state_bank import StageStateBank


def _state(seed: int) -> dict:
    torch.manual_seed(seed)
    return {
        "joint_pos": torch.randn(9),
        "joint_vel": torch.randn(9),
        "cube_2_pos": torch.randn(3), "cube_2_quat": torch.randn(4),
        "cube_2_lin_vel": torch.randn(3), "cube_2_ang_vel": torch.randn(3),
        "cube_1_pos": torch.randn(3), "cube_1_quat": torch.randn(4),
        "cube_1_lin_vel": torch.randn(3), "cube_1_ang_vel": torch.randn(3),
        "cube_3_pos": torch.randn(3), "cube_3_quat": torch.randn(4),
        "cube_3_lin_vel": torch.randn(3), "cube_3_ang_vel": torch.randn(3),
    }


def test_add_count_sample():
    bank = StageStateBank(max_capacity=5)
    bank.add("grasp", _state(1))
    bank.add("grasp", _state(2))
    bank.add("lift", _state(3))
    assert bank.count("grasp") == 2
    assert bank.count("lift") == 1
    assert bank.count() == 3
    s = bank.sample("grasp", torch.device("cpu"))
    assert s is not None and s["joint_pos"].shape == (9,)
    assert bank.sample("move", torch.device("cpu")) is None   # 空阶段 → None


def test_capacity_capped():
    bank = StageStateBank(max_capacity=2)
    for i in range(5):
        bank.add("grasp", _state(i))
    assert bank.count("grasp") == 2


def test_save_load_roundtrip(tmp_path):
    bank = StageStateBank(max_capacity=7)
    bank.add("grasp", _state(1))
    bank.add("grasp", _state(2))
    path = tmp_path / "grasp_bank.pt"
    bank.save(path)
    assert path.is_file()

    bank2 = StageStateBank()
    bank2.load(path)
    assert bank2.count("grasp") == 2
    assert bank2.max_capacity == 7
    s = bank2.sample("grasp", torch.device("cpu"))
    assert s is not None
