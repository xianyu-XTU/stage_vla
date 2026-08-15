"""pytest 公共配置：仓库根加入 sys.path + 合成状态 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings


@pytest.fixture(scope="session")
def settings():
    """解析后的配置（default + config.local.yaml 合并）。"""
    return load_settings()


@pytest.fixture
def synth_stack_state() -> dict:
    """4 个环境的合成堆叠状态（approach/grasp/lift/stack 各一）。"""
    N = 4
    ee = torch.tensor(
        [
            [0.5, 0.0, 0.15],  # env0: 远端 → approach
            [0.45, 0.0, 0.10],  # env1: 接近已抓住 → grasp
            [0.50, 0.0, 0.20],  # env2: 抓住并抬起 → lift
            [0.55, 0.0, 0.18],  # env3: 对齐并堆叠 → stack
        ],
        dtype=torch.float,
    )
    cubes = {
        "cube_1": torch.tensor([[0.55, 0.0, 0.02]] * N, dtype=torch.float),  # 底座（蓝）
        "cube_2": torch.tensor(  # 要抓（红）
            [[0.45, 0.0, 0.02], [0.45, 0.0, 0.04], [0.50, 0.0, 0.10], [0.55, 0.0, 0.08]],
            dtype=torch.float,
        ),
        "cube_3": torch.tensor([[0.5, -0.1, 0.02]] * N, dtype=torch.float),
    }
    grasp = torch.tensor([0.0, 1.0, 1.0, 1.0])
    stacked = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return {"ee": ee, "cubes": cubes, "grasp": grasp, "stacked": stacked}
