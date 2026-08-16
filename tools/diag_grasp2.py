"""diag_grasp2.py —— 改进抓取：全开接近 + 指尖对准方块中心 + 慢闭合 + 慢抬升。

基于实测（手指全开 80mm > 方块 50mm），关键是把末端高度调到让**指尖**对准方块
中心，而不是末端压进方块。逐步：测手指偏移 → 算接近高度 → 全开接近 → 慢闭 → 慢抬。
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
CUBE_H = 0.05  # 5cm 方块


def slow_move_ee_to(env, target, gripper, steps=80, tol=0.01, tag=""):
    """慢速 IK 移动：小增量（防把方块撞走 / 抬升猛拽）。"""
    inner = env.unwrapped
    dev = inner.device
    target = target.to(dev)
    for _ in range(steps):
        ee = inner.scene["ee_frame"].data.target_pos_w[:, 0, :]
        delta = (target - ee[0]).unsqueeze(0)
        if delta.norm().item() < tol:
            break
        act = torch.zeros(1, 7, device=dev)
        act[:, :3] = delta.clamp(-0.02, 0.02)   # 慢速
        act[:, 6] = gripper
        env.step(act)
    return delta.norm().item()


def main() -> int:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=96)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped
    dev = inner.device
    env.reset()

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    body_names = inner.scene["robot"].data.body_names
    lf_id = body_names.index("panda_leftfinger")
    rf_id = body_names.index("panda_rightfinger")
    ee_frame_idx = body_names.index("panda_hand")

    # 测手指相对末端框架的 z 偏移（指尖挂多低）
    ee_pos = inner.scene["robot"].data.body_pos_w[:, ee_frame_idx, :]
    lf_pos = inner.scene["robot"].data.body_pos_w[:, lf_id, :]
    rf_pos = inner.scene["robot"].data.body_pos_w[:, rf_id, :]
    finger_z_off = (ee_pos[0, 2] - (lf_pos[0, 2] + rf_pos[0, 2]) / 2).item()
    finger_gap = torch.norm(lf_pos[0] - rf_pos[0]).item()
    print(f"[g2] 手指相对末端 z 偏移 ≈ {finger_z_off*1000:.0f}mm（末端下方）", flush=True)
    print(f"[g2] 手指当前张距 = {finger_gap*1000:.0f}mm（方块 50mm）", flush=True)

    cube2 = _cube_pos(env, settings.task["cube_to_grasp"])
    cube_center_z = cube2[0, 2].item() + CUBE_H / 2
    print(f"[g2] 方块中心 z ≈ {cube_center_z:.3f}", flush=True)

    # 尝试几个接近高度（让指尖对准方块中心附近）
    for probe in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        env.reset()
        target = cube2 + torch.tensor([[0, 0, 0.15]], device=dev)
        move_ee_to(env, target, GRIPPER_OPEN, tag=f"pre-{probe}")
        ang = orient_ee_to_down(env, tag=f"orient-{probe}")
        # 接近高度 = 方块中心 + 手指偏移 + probe 微调
        ee_z_target = cube_center_z + finger_z_off + probe
        target = torch.tensor([[cube2[0, 0], cube2[0, 1], ee_z_target]], device=dev)
        move_ee_to(env, target, GRIPPER_OPEN, tag=f"approach-{probe}")
        # 慢闭合
        slow_move_ee_to(env, target, GRIPPER_CLOSE, steps=80, tag=f"close-{probe}")
        finger_j = inner.scene["robot"].data.joint_pos[0, 7].item()
        g1 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()
        # 慢抬升（上方 0.15）
        lift_target = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.15]], device=dev)
        slow_move_ee_to(env, lift_target, GRIPPER_CLOSE, steps=100, tag=f"lift-{probe}")
        g2 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()
        cube_z = _cube_pos(env, settings.task["cube_to_grasp"])[0, 2].item()
        print(f"[g2] probe={probe}: 手指joint={finger_j:.3f} 闭后grasped={g1:.2f} "
              f"抬升后grasped={g2:.2f} 方块z={cube_z:.3f} "
              f"({'✓ 夹稳并抬起!' if g2 > 0.5 and cube_z > 0.05 else '✗'})", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
