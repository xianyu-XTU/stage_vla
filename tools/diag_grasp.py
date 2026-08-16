"""diag_grasp.py —— 定向实验：找能让夹爪夹稳并抬起方块的高度（**需 Isaac 环境**）。

对每个抓取高度 h：接近→闭合→检查 grasped→抬升→再检查 grasped。
报告哪个高度能保持抓取（grasped 抬升后仍非零）。同时打印末端姿态（查是否手指朝下）。

用法::

    python tools\\run_isaaclab.py tools\\diag_grasp.py
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

from scripts.collect_primitive_data import _cube_pos, _ee_pos, move_ee_to  # noqa: E402
from stage_vla.vla_light.pose_utils import orient_ee_to_down  # noqa: E402

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0
HEIGHTS = [0.03, 0.04, 0.05]


def main() -> int:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=96)
    # 给目标方块加高摩擦（测试：手指压住但摩擦不够导致抬升滑脱）
    from isaaclab.envs.mdp import events
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.events.cube_high_friction = EventTermCfg(
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(settings.task["cube_to_grasp"]),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped
    dev = inner.device

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    cube2 = _cube_pos(env, settings.task["cube_to_grasp"])

    # 初始末端姿态
    quat = inner.scene["ee_frame"].data.target_quat_w[:, 0, :]
    print(f"[diag] 初始末端四元数(wxyz): {quat[0].tolist()}（z 分量≈-1 或 y≈±1 才手指朝下）", flush=True)

    for h in HEIGHTS:
        env.reset()
        # 预张开接近（高处）
        target = cube2 + torch.tensor([[0, 0, 0.12]], device=dev)
        move_ee_to(env, target, GRIPPER_OPEN, tag=f"pre@{h}")
        # 姿态对齐：手指朝下
        ang = orient_ee_to_down(env, tag=f"orient@{h}")
        # 下到抓取高度 → 闭合（长闭合时间，看手指能否到 0）
        target = cube2 + torch.tensor([[0, 0, h]], device=dev)
        move_ee_to(env, target, GRIPPER_OPEN, tag=f"approach@{h}")
        move_ee_to(env, target, GRIPPER_CLOSE, max_steps=60, tag=f"close@{h}")

        # 检查手指关节是否闭合（panda_finger_joint1/2，0=闭 0.04=开）
        finger = inner.scene["robot"].data.joint_pos[0, 7:9].tolist()
        print(f"[diag]   h={h:.2f} 手指关节={[round(f,3) for f in finger]} "
              f"({'✓ 已闭合' if finger[0] < 0.005 else '✗ 未闭合'})", flush=True)

        g1 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()

        # 抬升
        target = cube2 + torch.tensor([[0, 0, 0.12]], device=dev)
        move_ee_to(env, target, GRIPPER_CLOSE, tag=f"lift@{h}")
        g2 = mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()
        cube_z = _cube_pos(env, settings.task["cube_to_grasp"])[0, 2].item()
        print(f"[diag] h={h:.2f}: 姿态角={ang:.0f}° 闭合后 grasped={g1:.2f} | 抬升后 grasped={g2:.2f} | "
              f"方块z={cube_z:.3f} ({'✓ 夹稳' if g2 > 0.5 else '✗ 掉了'})", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
