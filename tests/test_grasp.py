"""真实抓取 / 项目 success 纯几何判定测试（交接文档 3/4/9 节）。"""

from __future__ import annotations

import torch

from stage_vla.stages import grasp as g


def _batch(*rows):
    return torch.tensor(list(rows), dtype=torch.float)


def test_cube_between_fingers_true():
    """方块在两指之间 → True（对侧夹持）。"""
    lf = _batch([0.45, -0.03, 0.10])
    rf = _batch([0.45, 0.03, 0.10])
    cube = _batch([0.45, 0.00, 0.10])
    assert bool(g.cube_between_fingers(lf, rf, cube)[0]) is True


def test_cube_between_fingers_same_side_false():
    """方块在两指同侧（都指向右侧）→ False。"""
    lf = _batch([0.45, -0.03, 0.10])
    rf = _batch([0.45, 0.03, 0.10])
    cube = _batch([0.45, 0.08, 0.10])   # 超过右指 → 投影不在缝隙内
    assert bool(g.cube_between_fingers(lf, rf, cube)[0]) is False


def test_cube_between_fingers_above_false():
    """方块架在指尖上方（离缝隙轴横向偏差大）→ False。"""
    lf = _batch([0.45, -0.03, 0.10])
    rf = _batch([0.45, 0.03, 0.10])
    cube = _batch([0.45, 0.00, 0.15])   # z 高 5cm → 离轴 5cm > perp_max 3cm
    assert bool(g.cube_between_fingers(lf, rf, cube, perp_max=0.03)[0]) is False


def test_fingers_closing():
    """手指部分闭合 → True；全开或一指开一指闭 → False。"""
    closed = _batch([0.025, 0.026])
    open_ = _batch([0.04, 0.04])
    mixed = _batch([0.04, 0.02])
    assert bool(g.fingers_closing(closed)[0]) is True
    assert bool(g.fingers_closing(open_)[0]) is False
    assert bool(g.fingers_closing(mixed)[0]) is False


def test_physical_grasp_geometric_hold_true():
    """真实持握（两指夹住 + 未全开 + 末端贴近）→ True。"""
    lf = _batch([0.45, -0.03, 0.10])
    rf = _batch([0.45, 0.03, 0.10])
    cube = _batch([0.45, 0.00, 0.10])
    ee = _batch([0.45, 0.00, 0.10])
    finger = _batch([0.025, 0.026])
    assert bool(g.physical_grasp_geometric(lf, rf, cube, ee, finger)[0]) is True


def test_physical_grasp_geometric_open_gripper_false():
    """张开的空手（未闭合）→ False——旧 `finger<0.01` 反例：空手全闭才算抓。"""
    lf = _batch([0.45, -0.03, 0.10])
    rf = _batch([0.45, 0.03, 0.10])
    cube = _batch([0.45, 0.00, 0.10])
    ee = _batch([0.45, 0.00, 0.10])
    finger = _batch([0.04, 0.04])   # 全开
    assert bool(g.physical_grasp_geometric(lf, rf, cube, ee, finger)[0]) is False


def test_physical_grasp_geometric_hand_empty_closed_false():
    """空手全闭（手指闭但方块不在两指间）→ False——交接文档 3 节核心反例。"""
    lf = _batch([0.45, -0.005, 0.10])
    rf = _batch([0.45, 0.005, 0.10])
    cube = _batch([0.50, 0.00, 0.10])   # 方块远离缝隙（没夹住）
    ee = _batch([0.45, 0.00, 0.10])
    finger = _batch([0.005, 0.005])     # 手指已闭死
    assert bool(g.physical_grasp_geometric(lf, rf, cube, ee, finger)[0]) is False


# ---------------------------------------------------------------------------
# 项目自定 success：红块在蓝块上
# ---------------------------------------------------------------------------
def test_red_on_blue_geometry_stacked_released_true():
    """红块在蓝块上 + 夹爪已释放 → True。"""
    lower = _batch([0.5, 0.0, 0.02])
    upper = _batch([0.5, 0.0, 0.07])          # 高 5cm ≈ cube_height
    finger = _batch([0.04, 0.04])             # 已打开（release）
    assert bool(g.red_on_blue_geometry(upper, lower, finger)[0]) is True


def test_red_on_blue_geometry_not_released_false():
    """几何堆叠但夹爪仍抓着（未释放）→ False——悬停夹持不是真堆叠。"""
    lower = _batch([0.5, 0.0, 0.02])
    upper = _batch([0.5, 0.0, 0.07])
    finger = _batch([0.02, 0.02])             # 仍闭合
    assert bool(g.red_on_blue_geometry(upper, lower, finger)[0]) is False


def test_red_on_blue_geometry_misaligned_false():
    """未对齐（水平差 > 阈值）→ False。"""
    lower = _batch([0.5, 0.0, 0.02])
    upper = _batch([0.5, 0.08, 0.07])
    finger = _batch([0.04, 0.04])
    assert bool(g.red_on_blue_geometry(upper, lower, finger)[0]) is False


def test_red_on_blue_geometry_wrong_height_false():
    """高度不对（悬在底座上方未落下）→ False。"""
    lower = _batch([0.5, 0.0, 0.02])
    upper = _batch([0.5, 0.0, 0.10])          # 高 8cm ≠ cube_height 5cm
    finger = _batch([0.04, 0.04])
    assert bool(g.red_on_blue_geometry(upper, lower, finger)[0]) is False


def test_red_on_blue_frame_velocity():
    """单帧 success：加低速条件——刚掉落还在晃/滑的假成功被排除。"""
    lower = _batch([0.5, 0.0, 0.02])
    upper = _batch([0.5, 0.0, 0.07])
    finger = _batch([0.04, 0.04])
    slow = _batch([0.0, 0.0, 0.01])
    fast = _batch([0.5, 0.0, 0.0])            # 高速滑动
    assert bool(g.red_on_blue_frame(upper, lower, finger, slow)[0]) is True
    assert bool(g.red_on_blue_frame(upper, lower, finger, fast)[0]) is False
