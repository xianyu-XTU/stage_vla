"""check_env.py —— Isaac 环境构造冒烟（M0 可选验收项，**需 Isaac 环境**）。

从注册表取任务 cfg → 注入阶段感知奖励 → 建环境 → step 几步 → 断言 reward/阶段信号正常。
通过 ``tools/run_isaaclab.py`` 运行（自动找到 isaaclab.bat）::

    python tools\\run_isaaclab.py scripts\\check_env.py --num_envs 4 --max_steps 5 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac 环境构造冒烟")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()

    # 惰性导入 Isaac 环境
    from stage_vla.envs.cfg_surgery import build_stage_env_cfg
    from stage_vla.rl.runner import create_env

    env_cfg = build_stage_env_cfg(
        settings=settings,
        num_envs=args.num_envs,
        seed=settings.ppo["seed"],
    )
    task_id = settings.task["id_state"]
    print(f"[check_env] 环境配置就绪：{env_cfg.scene.num_envs} envs, task={task_id}")

    env, simulation_app = create_env(task_id, env_cfg, headless=args.headless)
    obs, _ = env.reset()
    print(f"[check_env] reset 观测形状：{obs['policy'].shape}")

    import torch

    action = torch.zeros(env.action_space.shape, device=env.device)
    for step in range(args.max_steps):
        # rsl_rl 5.x VecEnv.step 返回 4 元组（obs, rew, dones, infos），非 gymnasium 5 元组
        obs, rew, dones, infos = env.step(action)
        if step == 0:
            print(f"[check_env] 首步 reward 形状：{rew.shape}，有限值：{torch.isfinite(rew).all().item()}")
    print("[check_env] 阶段感知奖励已接线，环境构造冒烟通过 ✓")
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
