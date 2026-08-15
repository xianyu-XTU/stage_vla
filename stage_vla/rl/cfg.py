"""cfg.py —— rsl_rl 5.x PPO 配置构造（**需 Isaac 环境**，惰性 import）。

从 ``config/default.yaml`` 的 ``ppo`` 节构造 ``RslRlOnPolicyRunnerCfg`` 子类，
并做 rsl_rl 5.x 废弃字段迁移（``handle_deprecated_rsl_rl_cfg``）。

字段严格对齐 Isaac Lab 3.0.0 + rsl-rl-lib 5.0.1 已验证写法（旧 E:\\stage_vla 跑通）：
- ``RslRlMLPModelCfg`` 用 ``hidden_dims=``/``activation=``（**无** ``network_cfg`` 字段）；
- 初始噪声在分布里：``RslRlMLPModelCfg.GaussianDistributionCfg(init_std=...)``，
  **不在** algorithm 里（``init_noise_std`` 为已废弃字段）；
- ``RslRlPpoAlgorithmCfg`` 必填 ``schedule`` / ``value_loss_coef`` / ``use_clipped_value_loss``；
- 只有 actor 挂 ``distribution_cfg``，critic 是确定性 MLP。
"""

from __future__ import annotations


def build_runner_cfg(ppo_cfg: dict):
    """从 config 的 ``ppo`` 节构造 rsl_rl 5.x PPO runner 配置。

    Args:
        ppo_cfg: config 的 ppo 节字典（含 seed/num_envs/…/actor_hidden_dims 等）

    Returns:
        ``RslRlOnPolicyRunnerCfg`` 子类实例（尚未做废弃字段迁移，由调用方 handle）。
    """
    from isaaclab.utils.configclass import configclass
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    @configclass
    class StackCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
        """堆叠任务阶段感知 PPO 训练配置。"""

        seed = ppo_cfg["seed"]
        device = "cuda:0"

        num_steps_per_env = ppo_cfg["num_steps_per_env"]
        max_iterations = ppo_cfg["max_iterations"]
        save_interval = ppo_cfg["save_interval"]

        experiment_name = ppo_cfg["experiment_name"]
        run_name = "stare_ppo"
        logger = "tensorboard"
        empirical_normalization = False

        # rsl_rl 5.x：actor/critic 各自的观测组
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}

        actor = RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=ppo_cfg["actor_hidden_dims"],
            activation=ppo_cfg["activation"],
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                class_name="GaussianDistribution",
                init_std=ppo_cfg["init_noise_std"],
                std_type="scalar",
            ),
        )
        critic = RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=ppo_cfg["critic_hidden_dims"],
            activation=ppo_cfg["activation"],
            obs_normalization=True,
        )

        algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=ppo_cfg["clip_param"],
            entropy_coef=ppo_cfg["entropy_coef"],
            num_learning_epochs=ppo_cfg["num_learning_epochs"],
            num_mini_batches=ppo_cfg["num_mini_batches"],
            learning_rate=ppo_cfg["learning_rate"],
            schedule="adaptive",
            gamma=ppo_cfg["gamma"],
            lam=ppo_cfg["lam"],
            desired_kl=ppo_cfg["desired_kl"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
        )

    return StackCubePPORunnerCfg()
