"""collect_primitive_data.py —— 脚本化阶段位姿数据采集（**需 Isaac 环境**）。

用 IK 反馈循环把机械臂末端摆到各基础动作（阶段）的几何目标位姿，逐阶段采集带标签
图像，得到**均衡**的 (图像, 阶段) 数据集（随机动作只会产生 approach，这里强制摆位）。

阶段 → 目标末端位置（相对目标方块 / 底座）：
    approach: 末端到 cube_2 上方（夹爪开）
    grasp:    末端在 cube_2 上方 + 夹爪闭合（抓住）
    lift:     末端抬升到 cube_2 上方 + 抬升高度（夹爪闭）
    move:     末端到 cube_1 上方 + 抬升高度（夹爪闭，搬运中）
    stack:    末端到 cube_1 上方 + 夹爪开（释放/堆叠）

数据保存：``outputs/primitive_data.pt`` → {"images": [N*K,H,W,3] uint8, "stages": [N*K]}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402
from stage_vla.vla_light.primitives import PRIMITIVE_NAMES  # noqa: E402

GRIPPER_OPEN = 1.0      # BinaryJointPositionAction：+1 开 / -1 闭
GRIPPER_CLOSE = -1.0
LIFT_DZ = 0.10          # 抬升高度（> lift_height 阈值）
REACH_Z = 0.10          # 末端到方块上方的高度


def _ee_pos(env) -> torch.Tensor:
    return env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]


def _cube_pos(env, name: str) -> torch.Tensor:
    return env.unwrapped.scene[name].data.root_pos_w


def move_ee_to(env, target: torch.Tensor, gripper: float, max_steps: int = 80, tol: float = 0.02,
               tag: str = "") -> None:
    """IK 反馈循环：末端移动到 target [3]（相对动作增量，逐步逼近）。"""
    inner = env.unwrapped
    dev = inner.device
    target = target.to(dev)
    N = 1
    for step in range(max_steps):
        ee = _ee_pos(env)                                          # [N,3]
        delta = (target - ee[0]).unsqueeze(0)                      # [1,3]
        dist = delta.norm().item()
        if step % 10 == 0:
            print(f"    [{tag}] step {step} dist={dist:.4f} ee={ee[0].tolist()}", flush=True)
        if dist < tol:
            break
        # 动作：[3 位置增量(限幅), 3 姿态 0, 夹爪]（IK-Rel 相对动作）
        act = torch.zeros(N, 7, device=dev)
        act[:, :3] = delta.clamp(-0.15, 0.15)
        act[:, 6] = gripper
        env.step(act)
    # 确保夹爪到位 + 稳定几帧
    for _ in range(2):
        act = torch.zeros(1, 7, device=dev)
        act[:, 6] = gripper
        env.step(act)


def collect(samples: int, cam_res: int = 96, seed: int = 0) -> dict:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=seed, cam_res=cam_res)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped

    images, stages = [], []
    for s in range(samples):
        obs, _ = env.reset()          # 随机化方块位置
        cube2 = _cube_pos(env, settings.task["cube_to_grasp"])   # [1,3]
        cube1 = _cube_pos(env, settings.task["cube_to_stack_on"])
        for st, name in enumerate(PRIMITIVE_NAMES):
            # 各阶段目标位置
            if name == "approach":
                target = cube2 + torch.tensor([[0, 0, REACH_Z]], device=inner.device)
                gripper = GRIPPER_OPEN
            elif name == "grasp":
                target = cube2 + torch.tensor([[0, 0, REACH_Z]], device=inner.device)
                gripper = GRIPPER_CLOSE
            elif name == "lift":
                target = cube2 + torch.tensor([[0, 0, REACH_Z + LIFT_DZ]], device=inner.device)
                gripper = GRIPPER_CLOSE
            elif name == "move":
                target = cube1 + torch.tensor([[0, 0, REACH_Z + LIFT_DZ]], device=inner.device)
                gripper = GRIPPER_CLOSE
            else:  # stack
                target = cube1 + torch.tensor([[0, 0, REACH_Z]], device=inner.device)
                gripper = GRIPPER_OPEN

            move_ee_to(env, target, gripper, tag=f"{name}@{s}")
            obs, *_ = env.step(torch.zeros(1, 7, device=inner.device))
            img = obs["policy"]["table_cam"][0].cpu().numpy()
            images.append(img)
            stages.append(st)

    data = {"images": np.stack(images), "stages": np.array(stages)}
    print(f"[collect] 采集完成：{len(images)} 张，开始保存", flush=True)
    # 保存必须在 close 之前（close 可能崩/挂，避免丢数据）
    out = _ROOT / "outputs" / "primitive_data.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out))
    print(f"[collect] 已保存 {out}", flush=True)
    env.close()
    sim_app.close()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="脚本化阶段位姿数据采集")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cam_res", type=int, default=96)
    parser.add_argument("--out", type=str, default="outputs/primitive_data.pt")
    args = parser.parse_args()

    print(f"[collect] 采集 {args.samples} 样本 × {len(PRIMITIVE_NAMES)} 阶段（cam_res={args.cam_res}）", flush=True)
    data = collect(args.samples, args.cam_res)

    out = _ROOT / args.out          # 绝对路径（避免 isaaclab.bat CWD 漂移）
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out))
    # 统计每阶段数量
    from collections import Counter
    dist = dict(sorted(Counter(data["stages"].tolist()).items()))
    print(f"[collect] 已保存 {out}：{len(data['images'])} 张图像，阶段分布 {dist}", flush=True)
    print(f"[collect] 图像形状 {data['images'].shape} dtype {data['images'].dtype}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
