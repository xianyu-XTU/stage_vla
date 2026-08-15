"""runner.py —— Isaac Lab 训练/评估运行器（**需 Isaac 环境**）。

封装 rsl_rl 5.x 的启动与训练循环：

    create_env → gym.make(task, cfg) → RslRlVecEnvWrapper
    build_agent_cfg → StackCubePPORunnerCfg + handle_deprecated_rsl_rl_cfg
    train → OnPolicyRunner.learn(...)

回归点（吸收旧工程教训）：
- rsl_rl 5.x 必须 ``obs_groups = {"actor": ["policy"], "critic": ["policy"]}``、
  ``actor/critic = RslRlMLPModelCfg``（含 distribution_cfg），训练前
  ``handle_deprecated_rsl_rl_cfg``。
- 观测组 ``observations.policy.concatenate_terms = True``。

本文件所有 Isaac 依赖都是**函数内惰性 import**，纯配置/无 Isaac 环境下不会因
``import stage_vla.rl.runner`` 而失败（但调用函数本身需要 Isaac 环境）。
"""

from __future__ import annotations

from pathlib import Path


def create_env(task_id: str, env_cfg, headless: bool = True):
    """启动 Isaac Sim Kit → 创建环境（gym.make + RslRlVecEnvWrapper）。

    Isaac Lab 3.0：``AppLauncher`` 构造即启动 SimulationApp（无 ``launch()`` 方法），
    app 经 ``app_launcher.app`` 访问。

    Returns:
        (env, simulation_app)：Isaac Lab VecEnv（经 rsl_rl wrapper）与 simulation app
        （调用方在结束时 ``simulation_app.close()``）。
    """
    import gymnasium as gym
    from isaaclab.app import AppLauncher
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    app_launcher = AppLauncher({"headless": headless})
    simulation_app = app_launcher.app

    env = gym.make(task_id, cfg=env_cfg)
    return RslRlVecEnvWrapper(env), simulation_app


def build_agent_cfg(ppo_cfg: dict, overrides: dict | None = None):
    """构造 rsl_rl 5.x PPO runner 配置并做废弃字段迁移。"""
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from .cfg import build_runner_cfg

    agent_cfg = build_runner_cfg(ppo_cfg)
    if overrides:
        for key, value in overrides.items():
            setattr(agent_cfg, key, value)
    handle_deprecated_rsl_rl_cfg(agent_cfg)
    return agent_cfg


def train(
    task_id: str,
    env_cfg,
    agent_cfg,
    max_iterations: int,
    headless: bool = True,
    log_root: Path | None = None,
):
    """执行 PPO 训练（OnPolicyRunner.learn）。"""
    from rsl_rl.runners import OnPolicyRunner

    env, simulation_app = create_env(task_id, env_cfg, headless=headless)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=str(log_root or "logs"))
    runner.learn(num_learning_iterations=max_iterations, init_at_random_episode=True)
    simulation_app.close()
    return runner


def load_policy(task_id: str, checkpoint: Path, env_cfg, agent_cfg, headless: bool = True):
    """加载训练 checkpoint，返回推理 policy。"""
    from rsl_rl.runners import OnPolicyRunner

    env, simulation_app = create_env(task_id, env_cfg, headless=headless)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=str(checkpoint.parent))
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    simulation_app.close()
    return policy
