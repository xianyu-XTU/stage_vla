"""diag_grasp5.py —— 自适应再对齐抓取（处理"下降推走方块"）。

假设：下降时手指可能撞到方块边缘把它推走，导致方块不在手指间。
流程：高位对齐+定向 → 慢降 → **重读方块当前位置** → 重新水平对准（紧容差）→ 慢闭 → 慢抬。
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
FINGER_Z_OFF = 0.058


def slow_move(env, target, gripper, max_steps=400, tol=0.004, step=0.004, tag=""):
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
    return inner.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()


def close_gripper(env, n_steps=100):
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

    def grasped():
        return mdp.object_grasped(
            env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        )[0].item()

    env.reset()
    cube2 = _cube_pos(env, settings.task["cube_to_grasp"])
    print(f"[g5] 初始方块: {cube2[0].tolist()}", flush=True)

    # 高位对齐 + 定向
    high = torch.tensor([[cube2[0, 0], cube2[0, 1], 0.25]], device=dev)
    move_ee_to(env, high, GRIPPER_OPEN, tag="high")
    orient_ee_to_down(env, tag="orient")

    # 慢降到手指到方块中心高度
    target_ee_z = cube2[0, 2].item() + 0.025 + FINGER_Z_OFF
    t1 = torch.tensor([[cube2[0, 0], cube2[0, 1], target_ee_z]], device=dev)
    slow_move(env, t1, GRIPPER_OPEN, tag="scoop")

    # 重读方块位置（可能被推）
    cube2_now = _cube_pos(env, settings.task["cube_to_grasp"])
    dx = (cube2_now[0, :2] - cube2[0, :2]).norm().item()
    print(f"[g5] 下降后方块: {cube2_now[0].tolist()} 位移={dx*1000:.1f}mm", flush=True)

    # 自适应再对齐到当前方块位置
    t2 = torch.tensor([[cube2_now[0, 0], cube2_now[0, 1], target_ee_z]], device=dev)
    ee_final = slow_move(env, t2, GRIPPER_OPEN, step=0.003, tag="realign")
    print(f"[g5] 再对齐后末端: {ee_final[0].tolist()} 方块: {_cube_pos(env, settings.task['cube_to_grasp'])[0].tolist()}", flush=True)

    # 慢闭 + 慢抬
    finger_j = close_gripper(env, n_steps=100)
    g1 = grasped()
    lift = torch.tensor([[_cube_pos(env, settings.task['cube_to_grasp'])[0, 0],
                         _cube_pos(env, settings.task['cube_to_grasp'])[0, 1], 0.25]], device=dev)
    slow_move(env, lift, GRIPPER_CLOSE, step=0.004, tag="lift")
    g2 = grasped()
    cube_z = _cube_pos(env, settings.task["cube_to_grasp"])[0, 2].item()
    held = g2 > 0.5 and cube_z > 0.05
    print(f"[g5] 闭合后手指={finger_j:.3f} g={g1:.2f} | 抬升后 g={g2:.2f} 方块z={cube_z:.3f} "
          f"({'✓ 夹稳抬起!' if held else '✗'})", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
