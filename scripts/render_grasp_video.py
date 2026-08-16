"""render_grasp_video.py —— 用训练好的策略 rollout，渲染抬起方块的视频（**需 Isaac 环境**）。

加载 rsl_rl checkpoint（model_2999.pt），在带相机的视觉环境里 rollout，
拼接状态 obs 喂给策略，抓取 table_cam 帧 → 合成 mp4（imageio_ffmpeg）。

用法::

    python tools\\run_isaaclab.py scripts\\render_grasp_video.py --ckpt logs\\model_2999.pt --steps 500
"""

from __future__ import annotations

import argparse
import subprocess
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
from stage_vla.rl.runner import build_agent_cfg  # noqa: E402

# 状态 obs 拼接顺序（与 id_state 的 94 维一致，排除图像）
STATE_KEYS = ["actions", "joint_pos", "joint_vel", "object", "cube_positions",
              "cube_orientations", "eef_pos", "eef_quat", "gripper_pos"]


def concat_state(obs_policy: dict, device) -> torch.Tensor:
    """把嵌套状态 obs 拼接成策略期望的扁平向量（94 维）。"""
    parts = []
    for k in STATE_KEYS:
        t = obs_policy[k]
        parts.append(torch.as_tensor(t).to(device).flatten(1))
    return torch.cat(parts, dim=1)


def frames_to_mp4(frames: list[np.ndarray], out: Path, fps: int = 15):
    """用 imageio_ffmpeg 的 ffmpeg 把帧数组编码成 mp4。"""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="logs/model_2999.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cam_res", type=int, default=128)
    parser.add_argument("--out", type=str, default="outputs/grasp_video.mp4")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    settings = load_settings()
    # 用状态版环境（策略训练时的环境，obs 直接匹配）+ 注入相机拍视频
    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from isaaclab.sensors import CameraCfg

    env_cfg = build_stage_env_cfg(settings, num_envs=1, seed=42)
    import isaaclab.sim as sim_utils
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=args.cam_res,
        width=args.cam_res,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 1.2), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"),
    )
    env_cfg.num_rerenders_on_reset = 3
    raw_env, sim_app = create_raw_env(settings.task["id_state"], env_cfg, headless=True, enable_cameras=True)
    inner = raw_env.unwrapped
    dev = inner.device

    # 套 wrapper（OnPolicyRunner 需要）+ 加载策略
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(raw_env)
    agent_cfg = build_agent_cfg(dict(settings.ppo))
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(_ROOT / "logs"))
    ckpt = _ROOT / args.ckpt
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=dev)
    print(f"[video] 已加载策略 {ckpt}，rollout {args.steps} 步", flush=True)

    # rollout 多个 episode，保存第一个抬起方块的（照官方 play.py 模式）
    best_frames, best_lift, best_ep = [], 0, -1
    for ep in range(10):
        env.reset()
        obs = env.get_observations()
        frames, cube_lift_count = [], 0
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs)
            obs, rew, dones, _ = env.step(action)
            policy.reset(dones)
            frame = inner.scene.sensors["table_cam"].data.output["rgb"][0].cpu().numpy()
            frames.append(frame)
            cube_z = inner.scene[settings.task["cube_to_grasp"]].data.root_pos_w[0, 2].item()
            if cube_z > 0.04:
                cube_lift_count += 1
            if bool(dones[0]):
                break
        print(f"[video] ep{ep}: 抬升 {cube_lift_count}/{len(frames)} 步", flush=True)
        if cube_lift_count > best_lift:
            best_frames, best_lift, best_ep = frames, cube_lift_count, ep
        if cube_lift_count > 50:   # 足够明显的抬升就停
            break

    print(f"[video] 最佳 ep{best_ep}：方块被抬起 {best_lift} 步", flush=True)
    if not best_frames:
        print("[video] 所有 episode 均未抬起方块", flush=True)
        env.close(); sim_app.close(); return 1
    out = _ROOT / args.out
    rc = frames_to_mp4(best_frames, out, fps=args.fps)
    print(f"[video] 已保存 {out}（{len(best_frames)} 帧，{rc}）", flush=True)
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
