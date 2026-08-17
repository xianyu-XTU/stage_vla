"""diag_red_on_blue.py —— 项目自定 success 的 scripted 冒烟验证（交接文档 2.4 节，**需 Isaac 环境**）。

不训练：手动把红块（cube_2）放到蓝块（cube_1）正上方、松开夹爪、稳定数步，
检查 ``red_on_blue_success`` 是否变 True。**若连这个都过不了，先别烧几千 iteration。**

用法::

    python tools\\run_isaaclab.py tools\\diag_red_on_blue.py --steps 120
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

CUBE_HEIGHT = 0.05
REST_HEIGHT = 0.0468   # 官方 cubes_stacked 的 height_diff：堆叠静止后两方块中心垂直距离


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = load_settings()
    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from stage_vla.rl.runner import create_raw_env

    env_cfg = build_stage_env_cfg(settings, num_envs=1, seed=args.seed)
    raw_env, sim_app = create_raw_env(settings.task["id_state"], env_cfg, headless=True)
    env = raw_env.unwrapped
    cube1 = env.scene["cube_1"]
    cube2 = env.scene["cube_2"]

    env.reset()
    # scripted 放置：红块中心放到蓝块中心正上方 REST_HEIGHT（官方 cubes_stacked 的
    # height_diff=0.0468，即堆叠静止后两中心垂直距离）。世界系，勿加 env_origins；
    # 零速度、xy 完全对齐——排除放置冲击，专注验证"叠好时 success 能否稳定触发"。
    base_pos = cube1.data.root_pos_w[0].clone()
    pose = torch.cat(
        [
            base_pos + torch.tensor([0.0, 0.0, REST_HEIGHT], device=env.device),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device),
        ]
    )
    cube2.write_root_pose_to_sim_index(root_pose=pose.unsqueeze(0), env_ids=torch.tensor([0], device=env.device))
    cube2.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([0], device=env.device)
    )

    robot = env.scene["robot"]
    reached = False
    for step in range(args.steps):
        action = torch.zeros(1, 7, device=env.device)     # 不动手臂；夹爪 0 = 打开（释放）
        env.step(action)
        # 逐项诊断：前 40 步每步打印，看几何/释放/速度哪项不满足、何时开始散开
        if step < 40 or step == args.steps - 1:
            finger = robot.data.joint_pos[0, 7:9]
            released = bool(torch.isclose(finger, torch.full_like(finger, 0.04), atol=1e-3).all())
            up, lo = cube2.data.root_pos_w[0], cube1.data.root_pos_w[0]
            h = float(torch.linalg.norm(up[2:] - lo[2:]))
            xy = float(torch.linalg.norm(up[:2] - lo[:2]))
            vel = float(cube2.data.root_lin_vel_w[0].norm())
            print(f"[diag] step {step}: cube2={[f'{v:.3f}' for v in up.tolist()]} "
                  f"cube1={[f'{v:.3f}' for v in lo.tolist()]} xy={xy:.4f} h={h:.4f} "
                  f"released={released} vel={vel:.3f} finger={[f'{v:.4f}' for v in finger.tolist()]}", flush=True)
        # 用 env 内部终止管理器写入的计数器判定（避免显式调用造成同帧双计）
        counter = env.extras.get("_stage_stack_counter")
        counter = counter if counter is not None else torch.zeros(1, dtype=torch.long, device=env.device)
        ok = int(counter[0]) >= 20
        if ok and not reached:
            print(f"[diag] ✅ red_on_blue_success（内部终止计数={int(counter[0])}）变 True @step {step}", flush=True)
            reached = True

    print(f"[diag] 结论：red_on_blue_success 在 {args.steps} 步内{'达到 ✅' if reached else '未达到 ❌'}", flush=True)
    if not reached:
        print("若未达到：检查 对齐/高度/夹爪释放/速度 阈值，以及 scripted 放置是否正确。", flush=True)
    env.close()
    sim_app.close()
    return 0 if reached else 1


if __name__ == "__main__":
    sys.exit(main())
