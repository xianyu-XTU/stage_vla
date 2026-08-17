"""train_stage.py —— 分阶段 RL：抓/抬/移动/堆叠 分开训练（**需 Isaac 环境**）。

把长程任务拆成 4 个独立的 RL 问题，每阶段单独训练一个策略（课程式）：
    --stage grasp  奖励=抓取到方块
    --stage lift   奖励=方块被抬起（官方机制）
    --stage move   奖励=方块靠近目标
    --stage stack  奖励=堆叠成功
每阶段独立 checkpoint 存 logs/stage_<name>/。

用法::

    python tools\\run_isaaclab.py scripts\\train_stage.py --stage lift --max_iterations 2000
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
from stage_vla.rl.runner import train  # noqa: E402

# 各阶段的奖励配置（奖励函数 + 权重）
STAGE_REWARDS = {
    "grasp": ("object_grasp_combined_reward", 5.0),   # 对侧定位 + 显式闭合（社区完整方案）
    "lift": ("object_is_lifted_reward", 15.0),
    "move": ("object_near_goal_reward", 5.0),
    "stack": ("object_stacked_dense_reward", 10.0),   # 稠密奖励（稀疏的学不会）
}


def build_stage_env_cfg(settings, stage: str, num_envs: int, seed: int):
    """按阶段构造环境：只给该阶段完成奖励（+接近塑形引导）。"""
    from isaaclab.envs import mdp
    from isaaclab.managers import RewardTermCfg as RewardTerm
    from isaaclab.utils.configclass import configclass
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from stage_vla.stages.rewards_isaac import (
        object_grasp_combined_reward,
        object_is_lifted_reward,
        object_near_goal_reward,
        object_stacked_dense_reward,
        stage_potential_reward,
    )

    env_cfg = load_cfg_from_registry(settings.task["id_state"], "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.observations.policy.concatenate_terms = True

    func_name, weight = STAGE_REWARDS[stage]
    stage_func = {
        "object_grasp_combined_reward": object_grasp_combined_reward,
        "object_is_lifted_reward": object_is_lifted_reward,
        "object_near_goal_reward": object_near_goal_reward,
        "object_stacked_dense_reward": object_stacked_dense_reward,
    }[func_name]

    @configclass
    class StageRewardsCfg:
        action_penalty = RewardTerm(func=mdp.action_l2, weight=settings.reward_weights["action_penalty"])
        stage_progress = RewardTerm(
            func=stage_potential_reward,
            params={"weights": settings.reward_weights, "gamma": 0.99},
            weight=settings.reward_weights["progress_shaping"],
        )
        stage_goal = RewardTerm(func=stage_func, weight=weight)

    env_cfg.rewards = StageRewardsCfg()
    return env_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="分阶段 RL 训练")
    parser.add_argument("--stage", choices=list(STAGE_REWARDS), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    env_cfg = build_stage_env_cfg(settings, args.stage, args.num_envs, settings.ppo["seed"])

    from stage_vla.rl.runner import build_agent_cfg

    agent_cfg = build_agent_cfg(dict(settings.ppo))
    log_root = _ROOT / "logs" / f"stage_{args.stage}"
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[train_stage] 阶段={args.stage} 奖励={STAGE_REWARDS[args.stage]} "
          f"{args.num_envs}env×{args.max_iterations}iter", flush=True)
    train(
        settings.task["id_state"], env_cfg, agent_cfg,
        args.max_iterations, headless=args.headless, log_root=log_root,
    )
    print(f"[train_stage] 完成，checkpoint 在 {log_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
