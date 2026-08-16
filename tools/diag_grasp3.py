"""diag_grasp3.py —— 慢速铲式接近抓取（避免下降撞开方块）。

从高处(z=0.25)以极小步长(±0.008)垂直下降，让张开的 80mm 手指像铲子一样顺着
方块两侧滑下包住它。测多个下降深度，找手指正好包住方块的位置 → 慢闭 → 慢抬。
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


def slow_step(env, target, gripper, max_steps=200, tol=0.008, step=0.008, tag=""):
    """极慢 IK 移动（每次增量 ±step），用于铲式下降/慢抬升。"""
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


def main() -> int:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=96)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped
    dev = inner.device

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    # 下降深度目标：手指尖端（ee-0.058）到达方块不同高度
    DESCENTS = {"上方": 0.055, "中心": 0.045, "下方": 0.035, "底部": 0.025}

    for name, finger_tip_z in DESCENTS.items():
        env.reset()
        cube2 = _cube_pos(env, settings.task["cube_to_grasp"])
        # 高位对齐 + 手指朝下
        high = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.25]], device=dev)
        move_ee_to(env, high, GRIPPER_OPEN, tag=f"high-{name}")
        orient_ee_to_down(env, tag=f"orient-{name}")
        # 铲式慢降：末端 z = 手指目标 + 0.058
        target_z = finger_tip_z + 0.058
        target = torch.tensor([[cube2[0, 0], cube2[0, 1], target_z]], device=dev)
        slow_step(env, target, GRIPPER_OPEN, tag=f"scoop-{name}")
        # 慢闭
        slow_step(env, target, GRIPPER_CLOSE, step=0.004, tag=f"close-{name}")
        finger_j = inner.scene["robot"].data.joint_pos[0, 7].item()
        g1 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()
        # 慢抬升
        lift = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.25]], device=dev)
        slow_step(env, lift, GRIPPER_CLOSE, tag=f"lift-{name}")
        g2 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()
        cube_z = _cube_pos(env, settings.task["cube_to_grasp"])[0, 2].item()
        held = g2 > 0.5 and cube_z > 0.05
        print(f"[g3] {name}(指尖z={finger_tip_z}): 手指joint={finger_j:.3f} 闭后g={g1:.2f} "
              f"抬后g={g2:.2f} 方块z={cube_z:.3f} ({'✓ 夹稳抬起!' if held else '✗'})", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
