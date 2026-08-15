"""stare_ppo.py —— StARe-PPO：阶段感知策略优化框架（模块②）。

v1（M0~M1）：状态版阶段感知 PPO。策略网络是 MLP（rsl_rl），观测为状态量，
奖励是阶段感知稠密奖励（``stage_potential_reward + stage_transition_reward``）。
入口 :func:`train_stare` 委托 :mod:`stage_vla.rl.runner` 启动 Isaac 训练。

v2（M2 预留，对应申请书"以 OpenVLA 为基础 + PPO 构建 StARe-PPO"）：
把 VLA 当作 actor 包进 PPO，即 "VLA-as-policy"。本文件预留：

- :class:`VLAAsPolicy` 抽象（``act(obs, instruction, image)``）；
- :func:`train_vla_in_loop` 融合训练入口（M2 实现）。

8GB 显存铁律：OpenVLA-7B（4-bit ≈ 4.1GB）与 Isaac 渲染不可共存，融合训练默认
``rl.vla.loop_mode=record_replay`` 或 TCP 分离进程，或 ``vla_backend=vision_only``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.logging import get_logger

logger = get_logger(__name__)


def train_stare(
    settings,
    instruction: str | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    headless: bool = True,
    stage_rewards: bool = True,
) -> object:
    """v1：状态版阶段感知 PPO 训练（**需 Isaac 环境**）。

    **语义分离真正驱动训练**：解析指令 → 语义计划决定
    - 目标方块（抓哪个、放哪个）→ 阶段检测/奖励盯对的方块；
    - 活动阶段（指令覆盖的子阶段）→ 未覆盖阶段不给完成奖。

    Args:
        settings: 解析后的配置（:class:`~stage_vla.core.config.Settings`）
        instruction: 语言指令（缺省用 config task.desc）
        num_envs / max_iterations: 覆盖 config 的 ppo 节
        headless: 无头模式
        stage_rewards: 是否注入阶段感知奖励（False = 任务默认奖励，作对照基线）

    Returns:
        OnPolicyRunner（训练完成后包含已保存 checkpoint）
    """
    from .runner import build_agent_cfg, train
    from ..stages.semantic import SemanticSeparator, plan_targets

    instruction = instruction or settings.task["desc"]
    plan = SemanticSeparator().parse(instruction)
    cube_grasp, cube_stack, active_stages = plan_targets(
        plan,
        default_grasp=settings.task["cube_to_grasp"],
        default_stack=settings.task["cube_to_stack_on"],
    )

    ppo_cfg = dict(settings.ppo)
    if num_envs is not None:
        ppo_cfg["num_envs"] = num_envs
    if max_iterations is not None:
        ppo_cfg["max_iterations"] = max_iterations

    env_cfg = _build_env_cfg(
        settings,
        ppo_cfg["num_envs"],
        cube_to_grasp=cube_grasp,
        cube_to_stack_on=cube_stack,
        active_stages=active_stages,
        stage_rewards=stage_rewards,
    )
    agent_cfg = build_agent_cfg(ppo_cfg)

    logger.info(
        "StARe-PPO v1 训练启动：%d 环境 / %d 迭代 | 指令=%r 目标=(%s→%s) 活动阶段=%s 阶段奖励=%s",
        ppo_cfg["num_envs"], ppo_cfg["max_iterations"],
        instruction, cube_grasp, cube_stack, active_stages, stage_rewards,
    )
    return train(
        settings.task["id_state"],
        env_cfg,
        agent_cfg,
        ppo_cfg["max_iterations"],
        headless=headless,
    )


def _build_env_cfg(settings, num_envs: int, *, cube_to_grasp=None, cube_to_stack_on=None, active_stages=None, stage_rewards=True):
    """构造带阶段感知奖励的环境配置（需 Isaac 环境，惰性 import）。"""
    from ..envs.cfg_surgery import build_stage_env_cfg

    return build_stage_env_cfg(
        settings=settings,
        num_envs=num_envs,
        seed=settings.ppo["seed"],
        cube_to_grasp=cube_to_grasp,
        cube_to_stack_on=cube_to_stack_on,
        active_stages=active_stages,
        stage_rewards=stage_rewards,
    )


# ============================================================================
# v2 预留：VLA-as-policy 融合
# ============================================================================
class VLAAsPolicy(ABC):
    """把 VLA 策略包装成可被 PPO 当作 actor 的网络（M2 实现）。

    ``act`` 输出动作分布/采样动作；融合训练时还需提供可微动作头与损失回传接口。
    """

    @abstractmethod
    def act(self, obs, instruction: str, image) -> object:
        """给定观测/指令/图像，返回动作采样与分布信息。"""
        raise NotImplementedError


def train_vla_in_loop(settings, **overrides) -> object:
    """v2：VLA-as-policy 的融合训练入口（M2 实现）。

    目前未实现，抛出指引性错误，避免误用。
    """
    raise NotImplementedError(
        "StARe-PPO v2（VLA-as-policy 融合训练）为 M2 里程碑。当前可用 v1：train_stare()。"
    )
