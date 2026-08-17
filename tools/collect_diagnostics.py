"""collect_diagnostics.py —— 阶段诊断指标收集（交接文档 16 节，**需 Isaac 环境**）。

跑一个已训练策略，按 episode 收集并汇总诊断指标，回答"reward 高但 success=0
到底死在哪一阶段"。指标见 ``stage_vla.rl.diagnostics.StageMetrics``。

用法::

    python tools\\run_isaaclab.py tools\\collect_diagnostics.py --stage grasp --episodes 5 --steps 600
    python tools\\run_isaaclab.py tools\\collect_diagnostics.py --checkpoint logs\\stage_grasp\\model_2000.pt
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
from stage_vla.rl.diagnostics import StageMetrics  # noqa: E402


def load_policy(env, agent_cfg, stage: str | None, checkpoint: str | None, device: str):
    """加载已训练策略（指定 checkpoint，或某阶段 logs/stage_<stage> 最优 ckpt）。"""
    from rsl_rl.runners import OnPolicyRunner

    if checkpoint is None:
        run_dir = _ROOT / "logs" / f"stage_{stage}"
        ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if not ckpts:
            print(f"[diag] 无 {stage} checkpoint: {run_dir}", flush=True)
            sys.exit(1)
        checkpoint = str(ckpts[-1])
        print(f"[diag] 加载 {stage} 策略 {Path(checkpoint).name}", flush=True)
    else:
        print(f"[diag] 加载 checkpoint {checkpoint}", flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(Path(checkpoint).parent))
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["grasp", "lift", "move", "stack"], default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stable_steps", type=int, default=20)
    args = parser.parse_args()
    if args.stage is None and args.checkpoint is None:
        parser.error("必须提供 --stage 或 --checkpoint")

    settings = load_settings()

    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from stage_vla.rl.runner import build_agent_cfg, create_raw_env
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from stage_vla.stages import grasp as grasp_geo
    from stage_vla.stages.rewards_isaac import _physical_grasp, red_on_blue_success

    env_cfg = build_stage_env_cfg(settings, num_envs=1, seed=args.seed)
    raw_env, sim_app = create_raw_env(settings.task["id_state"], env_cfg, headless=True)
    inner = raw_env.unwrapped
    dev = inner.device
    env = RslRlVecEnvWrapper(raw_env)

    agent_cfg = build_agent_cfg(dict(settings.ppo))
    policy = load_policy(env, agent_cfg, args.stage, args.checkpoint, dev)

    metrics = StageMetrics(stable_steps=args.stable_steps)
    cube2 = inner.scene["cube_2"]
    cube1 = inner.scene["cube_1"]
    robot = inner.scene["robot"]

    for ep in range(args.episodes):
        env.reset()
        obs = env.get_observations()   # RslRlVecEnvWrapper.reset 返回 (obs, info)，不能直接喂 policy
        policy.reset()
        metrics.reset()
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs)
            obs, rew, dones, _ = env.step(action)
            policy.reset(dones)

            finger = robot.data.joint_pos[0, 7:9]
            released = bool(torch.isclose(finger, torch.full_like(finger, 0.04), atol=1e-3).all())
            metrics.record(
                stage=_stage_of(inner),
                physical=bool(_physical_grasp(inner)[0]),
                cube_z=float(cube2.data.root_pos_w[0, 2]),
                cube_vel=cube2.data.root_lin_vel_w[0],
                released=released,
                rb_frame=bool(
                    grasp_geo.red_on_blue_frame(
                        cube2.data.root_pos_w[0], cube1.data.root_pos_w[0],
                        finger, cube2.data.root_lin_vel_w[0],
                    )
                ),
            )
            if bool(dones[0]):
                break
        # 是否命中项目 success（终止步状态仍满足条件，重算即得同值）
        success = bool(red_on_blue_success(inner)[0])
        metrics.end_episode(success)
        print(f"[diag] episode {ep + 1}/{args.episodes} 结束 @step {step + 1} "
              f"success={success}", flush=True)

    metrics.print_report()
    env.close()
    sim_app.close()
    return 0


def _stage_of(env) -> int:
    """读取当前阶段索引（状态版检测器，detect_stage_from_env 的轻封装）。"""
    from stage_vla.stages.rewards_isaac import detect_stage_from_env

    return int(detect_stage_from_env(env)[0])


if __name__ == "__main__":
    sys.exit(main())
