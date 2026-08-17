"""run_staged_pipeline.py —— 组合执行：串联已训练的分阶段策略（**需 Isaac 环境**）。

按检测到的当前阶段，动态切换使用对应阶段的策略（grasp→lift→move→stack），
执行完整的"抓-抬-移动-堆叠"序列，记录每阶段成功。

用法::

    python tools\\run_isaaclab.py scripts\\run_staged_pipeline.py --steps 800 --cam_res 128
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
from stage_vla.rl.runner import build_agent_cfg  # noqa: E402


def load_stage_policy(env, agent_cfg, stage: str, device):
    """从 logs/stage_<stage>/ 加载最优 checkpoint，返回推理策略。"""
    from rsl_rl.runners import OnPolicyRunner

    run_dir = _ROOT / "logs" / f"stage_{stage}"
    # 按迭代号自然排序（字符串排序 model_9 > model_20，会选错）
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        print(f"[pipe] 无 {stage} checkpoint: {run_dir}", flush=True)
        return None
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir))
    runner.load(str(ckpts[-1]))
    print(f"[pipe] 加载 {stage} 策略 {ckpts[-1].name}（iter {ckpts[-1].stem}）", flush=True)
    return runner.get_inference_policy(device=device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--cam_res", type=int, default=128)
    parser.add_argument("--out", type=str, default="outputs/staged_pipeline.mp4")
    args = parser.parse_args()

    settings = load_settings()
    # 状态 env + 相机（组合执行 + 录像）
    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    env_cfg = build_stage_env_cfg(settings, num_envs=1, seed=42)
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam", update_period=0.0,
        height=args.cam_res, width=args.cam_res, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 2)),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.2), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"),
    )
    env_cfg.num_rerenders_on_reset = 3
    from stage_vla.rl.runner import create_raw_env
    raw_env, sim_app = create_raw_env(settings.task["id_state"], env_cfg, headless=True, enable_cameras=True)
    inner = raw_env.unwrapped
    dev = inner.device
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(raw_env)

    agent_cfg = build_agent_cfg(dict(settings.ppo))
    policies = {s: load_stage_policy(env, agent_cfg, s, dev) for s in ["grasp", "lift", "move", "stack"]}

    from stage_vla.stages.rewards_isaac import detect_stage_from_env

    env.reset()
    obs = env.get_observations()
    frames, stage_history = [], []
    policy_key = "grasp"          # 从 grasp 策略开始
    hold = 0                      # 当前阶段保持计数（防瞬时信号误切换）
    for step in range(args.steps):
        # 检测当前阶段，据此切换策略（grasp 0-1 / lift 1-2 / move 2-3 / stack 3-4）
        stage = int(detect_stage_from_env(inner)[0].item())
        if stage >= 3 and policy_key != "stack":
            policy_key = "stack"; hold = 0
            print(f"[pipe] 进入 stack 阶段 @{step}", flush=True)
        elif stage >= 2 and policy_key in ("grasp", "lift"):
            policy_key = "move"; hold = 0
            print(f"[pipe] 进入 move 阶段 @{step}", flush=True)
        elif stage >= 1 and policy_key == "grasp":
            hold += 1
            if hold >= 20:        # 稳定抓到 20 步才切 lift（防瞬时 grasp 误切）
                policy_key = "lift"; hold = 0
                print(f"[pipe] 进入 lift 阶段 @{step}（稳定抓到{hold}步）", flush=True)
        else:
            hold = 0

        policy = policies.get(policy_key)
        if policy is None:
            print(f"[pipe] 策略缺失，停止 @{step}", flush=True)
            break
        with torch.inference_mode():
            action = policy(obs)
        obs, rew, dones, _ = env.step(action)
        policy.reset(dones)
        stage_history.append(stage)
        frame = inner.scene.sensors["table_cam"].data.output["rgb"][0].cpu().numpy()
        frames.append(frame)
        if bool(dones[0]):
            print(f"[pipe] episode 终止 @{step}", flush=True)
            break

    # 统计
    import collections
    dist = dict(sorted(collections.Counter(stage_history).items()))
    max_stage = max(stage_history) if stage_history else 0
    print(f"[pipe] 到达最高阶段={['approach','grasp','lift','move','stack'][max_stage]} 阶段分布={dist}", flush=True)

    # 录像
    if frames:
        import subprocess
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        h, w = frames[0].shape[:2]
        out = _ROOT / args.out
        proc = subprocess.Popen([ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                 "-r", "15", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in frames:
            proc.stdin.write(f.astype(np.uint8).tobytes())
        proc.stdin.close(); proc.wait()
        print(f"[pipe] 已存视频 {out}", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
