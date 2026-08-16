"""diag_grasp4.py —— 最终抓取尝试（机制已清楚：方块必须在手指之间）。

正确几何：手指在末端下方 58mm、全开 80mm。末端 z 应 ≈ 方块中心(0.045)+0.058=0.103，
让张开的 80mm 手指包住 50mm 方块。
流程：高位对齐+定向 → 慢速下降(避开撞方块) → 独立慢闭合(修 bug) → 慢抬升。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402

from scripts.collect_primitive_data import _cube_pos, move_ee_to  # noqa: E402
from stage_vla.vla_light.pose_utils import orient_ee_to_down  # noqa: E402

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0
FINGER_Z_OFF = 0.058   # 手指相对末端下方距离（实测）


def slow_move(env, target, gripper, max_steps=300, tol=0.005, step=0.005, tag=""):
    """慢速 IK 移动：每次增量 ±step。返回最终末端 z。"""
    inner = env.unwrapped
    dev = inner.device
    target = target.to(dev)
    for _ in range(max_steps):
        ee = inner.scene["ee_frame"].data.target_pos_w[:, 0, :]
        delta = (target - ee[0]).unsqueeze(0)
        if delta.norm().item() < tol:
            break
        act = torch.zeros(1, 7, device=dev)
        act[:, :3] = delta.clamp(-step, step)
        act[:, 6] = gripper
        env.step(act)
    return inner.scene["ee_frame"].data.target_pos_w[:, 0, 2].item()


def close_gripper(env, n_steps=80):
    """独立慢闭合：只发夹爪命令，不看位置收敛（修 diag3 的 bug）。"""
    dev = env.unwrapped.device
    act = torch.zeros(1, 7, device=dev)
    act[:, 6] = GRIPPER_CLOSE
    for _ in range(n_steps):
        env.step(act)
    return env.unwrapped.scene["robot"].data.joint_pos[0, 7].item()


def main() -> int:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=96)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped
    dev = inner.device

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    env.reset()
    cube2 = _cube_pos(env, settings.task["cube_to_grasp"])
    cube_center_z = cube2[0, 2].item() + 0.025
    target_ee_z = cube_center_z + FINGER_Z_OFF
    print(f"[g4] 方块中心 z={cube_center_z:.3f}  目标末端 z={target_ee_z:.3f}", flush=True)

    # 1) 高位对齐 + 手指朝下
    high = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.25]], device=dev)
    move_ee_to(env, high, GRIPPER_OPEN, tag="high")
    orient_ee_to_down(env, tag="orient")
    print(f"[g4] 高位对齐 + 定向完成，末端 z={inner.scene['ee_frame'].data.target_pos_w[:,0,2].item():.3f}", flush=True)

    # 2) 慢速下降（手指开 80mm，包向方块）
    target = torch.tensor([[cube2[0, 0], cube2[0, 1], target_ee_z]], device=dev)
    end_z = slow_move(env, target, GRIPPER_OPEN, step=0.005, tag="scoop-down")
    finger_gap_open = torch.norm(
        inner.scene["robot"].data.body_pos_w[:, inner.scene["robot"].data.body_names.index("panda_leftfinger")]
        - inner.scene["robot"].data.body_pos_w[:, inner.scene["robot"].data.body_names.index("panda_rightfinger")]
    ).item()
    print(f"[g4] 下降后末端 z={end_z:.3f} 手指张距={finger_gap_open*1000:.0f}mm", flush=True)

    # 3) 独立慢闭合
    finger_j = close_gripper(env, n_steps=100)
    g1 = mdp.object_grasped(
        env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
    )[0].item()
    print(f"[g4] 闭合后手指joint={finger_j:.3f} grasped={g1:.2f}", flush=True)

    # 4) 慢抬升
    lift = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.25]], device=dev)
    slow_move(env, lift, GRIPPER_CLOSE, step=0.005, tag="slow-lift")
    g2 = mdp.object_grasped(
        env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
    )[0].item()
    cube_z = _cube_pos(env, settings.task["cube_to_grasp"])[0, 2].item()
    held = g2 > 0.5 and cube_z > 0.05
    print(f"[g4] 抬升后 grasped={g2:.2f} 方块 z={cube_z:.3f} ({'✓ 夹稳抬起!' if held else '✗'})", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
