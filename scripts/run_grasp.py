"""run_grasp.py —— 实测：脚本化基础动作执行器能否真正抓取桌面方块（**需 Isaac 环境**）。

按语义计划 [approach, grasp, lift, move, stack] 用 IK 控制器逐步执行基础动作，
每步用 StageDetector 检查阶段是否推进、object_grasped/stacked 是否触发。

用法::

    python tools\\run_isaaclab.py scripts\\run_grasp.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402
from stage_vla.vla_light.planner import semantic_plan  # noqa: E402

from scripts.collect_primitive_data import (  # noqa: E402
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LIFT_DZ,
    REACH_Z,
    _cube_pos,
    _ee_pos,
    move_ee_to,
)

GRASP_OK = "grasp_success"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--cam_res", type=int, default=96)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    instruction = args.instruction or settings.task["desc"]
    plan = semantic_plan(instruction)
    print(f"[grasp] 指令: {instruction}\n[grasp] 基础动作计划: {plan}", flush=True)

    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=args.cam_res)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=args.headless, enable_cameras=True)
    inner = env.unwrapped

    from stage_vla.stages.rewards_isaac import detect_stage_from_env
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp

    dev = inner.device
    obs, _ = env.reset()
    cube2 = _cube_pos(env, settings.task["cube_to_grasp"])
    cube1 = _cube_pos(env, settings.task["cube_to_stack_on"])

    # 抓取高度：方块在桌面(z≈0.02)，立方体高约 0.05，末端需贴到方块上方才触发 object_grasped
    GRASP_Z = 0.04

    # 执行计划：每步 move_ee_to 到位 + 夹爪 + 检查阶段
    for step, name in enumerate(plan):
        if name == "approach":
            target = cube2 + torch.tensor([[0, 0, GRASP_Z]], device=dev)
            gripper = GRIPPER_OPEN
        elif name == "grasp":
            target = cube2 + torch.tensor([[0, 0, GRASP_Z]], device=dev)
            gripper = GRIPPER_CLOSE
        elif name == "lift":
            target = cube2 + torch.tensor([[0, 0, GRASP_Z + LIFT_DZ]], device=dev)
            gripper = GRIPPER_CLOSE
        elif name == "move":
            target = cube1 + torch.tensor([[0, 0, GRASP_Z + LIFT_DZ]], device=dev)
            gripper = GRIPPER_CLOSE
        else:  # stack
            target = cube1 + torch.tensor([[0, 0, GRASP_Z]], device=dev)
            gripper = GRIPPER_OPEN

        move_ee_to(env, target, gripper, tag=f"{name}@{step}")
        stage = detect_stage_from_env(env)[0].item()
        print(f"[grasp] 执行 {name:<9} → 检测阶段={['approach','grasp','lift','move','stack'][int(stage)]}", flush=True)

    # 最终判定：是否抓到 + 是否堆叠
    obs, *_ = env.step(torch.zeros(1, 7, device=dev))
    grasp = mdp.object_grasped(
        env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
    )
    stacked = mdp.object_stacked(
        env.unwrapped, robot_cfg=SceneEntityCfg("robot"),
        upper_object_cfg=SceneEntityCfg(settings.task["cube_to_grasp"]),
        lower_object_cfg=SceneEntityCfg(settings.task["cube_to_stack_on"]),
    )
    print(f"\n[grasp] 最终: object_grasped={grasp[0].item()}  object_stacked={stacked[0].item()}", flush=True)
    if grasp[0].item() > 0.5:
        print("[grasp] ✓ 抓取成功！", flush=True)
    else:
        print("[grasp] ✗ 未抓取成功（末端位置/夹爪可能没对准方块）", flush=True)
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
