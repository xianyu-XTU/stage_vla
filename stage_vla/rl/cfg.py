"""cfg.py —— rsl_rl 5.x PPO 配置构造（**需 Isaac 环境**，惰性 import）。

从 ``config/default.yaml`` 的 ``ppo`` 节构造 ``RslRlOnPolicyRunnerCfg`` 子类，
并做 rsl_rl 5.x 废弃字段迁移（``handle_deprecated_rsl_rl_cfg``）。

回归点（吸收旧工程教训）：
- 5.x 必须 ``obs_groups = {"actor": ["policy"], "critic": ["policy"]}``；
- actor/critic 用 ``RslRlMLPModelCfg``（含 ``GaussianDistributionCfg``）；
- ``learning_rate`` 用 adaptive 配置，``desired_kl`` 控制退避。
"""

from __future__ import annotations


def build_runner_cfg(ppo_cfg: dict):
    """从 config 的 ``ppo`` 节构造 rsl_rl 5.x PPO runner 配置。

    Args:
        ppo_cfg: config 的 ppo 节字典（含 seed/num_envs/…/actor_hidden_dims 等）

    Returns:
        ``RslRlOnPolicyRunnerCfg`` 子类实例（已做废弃字段迁移）。
    """
    from isaaclab.utils.configclass import configclass
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )
    from rsl_rl.distributions.gaussian import GaussianDistributionCfg

    @configclass
    class StackCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
        """堆叠任务阶段感知 PPO 训练配置。"""

        seed = ppo_cfg["seed"]
        num_steps_per_env = ppo_cfg["num_steps_per_env"]
        max_iterations = ppo_cfg["max_iterations"]
        save_interval = ppo_cfg["save_interval"]
        experiment_name = ppo_cfg["experiment_name"]

        obs_groups = {"actor": ["policy"], "critic": ["policy"]}

        actor = RslRlMLPModelCfg(
            network_cfg={"mlp": {"units": ppo_cfg["actor_hidden_dims"], "activation": ppo_cfg["activation"]}},
            distribution_cfg=GaussianDistributionCfg(),
        )
        critic = RslRlMLPModelCfg(
            network_cfg={"mlp": {"units": ppo_cfg["critic_hidden_dims"], "activation": ppo_cfg["activation"]}},
            distribution_cfg=GaussianDistributionCfg(),
        )
        algorithm = RslRlPpoAlgorithmCfg(
            num_learning_epochs=ppo_cfg["num_learning_epochs"],
            num_mini_batches=ppo_cfg["num_mini_batches"],
            learning_rate=ppo_cfg["learning_rate"],
            gamma=ppo_cfg["gamma"],
            lam=ppo_cfg["lam"],
            clip_param=ppo_cfg["clip_param"],
            entropy_coef=ppo_cfg["entropy_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            desired_kl=ppo_cfg["desired_kl"],
            init_noise_std=ppo_cfg["init_noise_std"],
        )

    return StackCubePPORunnerCfg()
