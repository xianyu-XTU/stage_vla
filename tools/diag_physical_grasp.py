"""diag_physical_grasp.py —— 验证物理抓取几何判定能否被脚本化抓取触发（**需 Isaac 环境**）。

背景：旧 diag 显示任何 policy 的 physical_grasp 恒 0，根因是 ``body_pos_w`` 的
``panda_leftfinger/rightfinger`` 是手指**根/枢轴**（近手心），"方块在两指之间"的
几何判定对着错误位置算。已改用 FrameTransformer 的指尖帧（tool_leftfinger/rightfinger）。

本脚本做 scripted 抓取（接近→闭合→抬升），逐段打印指尖/方块/夹爪几何，
验证 ``cube_between_fingers`` 与 ``_physical_grasp`` 在真实抓取时能否变 True。

用法::

    python tools\\run_isaaclab.py tools\\diag_physical_grasp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402

GRASP_Z = 0.04        # 末端在方块上方的高度
LIFT_DZ = 0.10        # 抬升高度
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


def main() -> int:
    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from stage_vla.rl.runner import create_raw_env
    from stage_vla.stages import grasp as grasp_geo
    from stage_vla.stages.rewards_isaac import _physical_grasp

    settings = load_settings()
    env_cfg = build_stage_env_cfg(settings, num_envs=1, seed=42)
    raw_env, sim_app = create_raw_env(settings.task["id_state"], env_cfg, headless=True)
    env = raw_env.unwrapped
    dev = env.device

    from scripts.collect_primitive_data import _cube_pos, _ee_pos, move_ee_to

    raw_env.reset()                      # gymnasium wrapper 需先 reset（move_ee_to 步进它）
    cube2 = _cube_pos(raw_env, settings.task["cube_to_grasp"])

    def report(tag: str, step: int) -> None:
        frames = env.scene["ee_frame"].data.target_pos_w[0]     # [3,3]: ee/right/left
        lf, rf, ee = frames[2], frames[1], frames[0]
        finger = env.scene["robot"].data.joint_pos[0, 7:9]
        cube = cube2[0]
        between = bool(grasp_geo.cube_between_fingers(lf.unsqueeze(0), rf.unsqueeze(0), cube.unsqueeze(0))[0])
        closing = bool(grasp_geo.fingers_closing(finger.unsqueeze(0))[0])
        phy = bool(_physical_grasp(env)[0])
        print(
            f"  [{tag}] step {step}: lf={[f'{v:.3f}' for v in lf.tolist()]} "
            f"rf={[f'{v:.3f}' for v in rf.tolist()]} cube={[f'{v:.3f}' for v in cube.tolist()]} "
            f"finger={[f'{v:.4f}' for v in finger.tolist()]} "
            f"between={between} closing={closing} physical={phy}", flush=True,
        )

    # 1) 自适应下降：边降边查 cube_between_fingers，直到指尖真正环绕方块（再闭合）
    print("[diag] 接近并自适应下降…", flush=True)
    move_ee_to(raw_env, cube2 + torch.tensor([[0, 0, GRASP_Z]], device=dev), GRIPPER_OPEN, tag="approach")
    z = GRASP_Z
    reached_between = False
    while z > -0.02:
        move_ee_to(raw_env, cube2 + torch.tensor([[0, 0, z]], device=dev), GRIPPER_OPEN, tag=f"desc z={z:.2f}")
        frames = env.scene["ee_frame"].data.target_pos_w[0]
        between = bool(grasp_geo.cube_between_fingers(
            frames[2].unsqueeze(0), frames[1].unsqueeze(0), cube2[0].unsqueeze(0))[0])
        if between:
            reached_between = True
            print(f"  [diag] ✅ 指尖环绕方块 @z={z:.2f}", flush=True)
            break
        z -= 0.01
    report(f"desc end(z={z:.2f})", 0)
    # 2) 闭合（夹爪 -1=闭）
    print("[diag] 闭合…", flush=True)
    move_ee_to(raw_env, cube2 + torch.tensor([[0, 0, z]], device=dev), GRIPPER_CLOSE, tag="close")
    report("close", 0)
    # 3) 抬升
    print("[diag] 抬升…", flush=True)
    move_ee_to(raw_env, cube2 + torch.tensor([[0, 0, z + LIFT_DZ]], device=dev), GRIPPER_CLOSE, tag="lift")
    report("lift", 0)

    phy = bool(_physical_grasp(env)[0])
    print(f"\n[diag] 结论：物理抓取 _physical_grasp = {phy}", flush=True)
    if not phy:
        print("未触发：检查 指尖帧索引 / between 阈值(perp_max) / 抓取高度(GRASP_Z)。", flush=True)
    env.close()
    sim_app.close()
    return 0 if phy else 1


if __name__ == "__main__":
    sys.exit(main())
