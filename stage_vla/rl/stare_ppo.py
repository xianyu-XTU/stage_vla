"""stare_ppo.py —— StARe-PPO：阶段感知策略优化框架（模块②）。

v1（M0~M1）：状态版阶段感知 PPO。策略网络是 MLP（rsl_rl），观测为状态量，
奖励是阶段感知稠密奖励（``stage_potential_reward + stage_transition_reward``）。
入口 :func:`train_stare` 委托 :mod:`stage_vla.rl.runner` 启动 Isaac 训练。

v2（M2）：把 VLA 当作 actor 包进 PPO，即 "VLA-as-policy"。接口实现在
:mod:`stage_vla.rl.vla_policy`，融合训练循环在 :mod:`stage_vla.rl.ppo_loop`
（:func:`train_vla_in_loop`）。本文件提供 v1 状态版训练（:func:`train_stare`）。

8GB 显存铁律：OpenVLA-7B（4-bit ≈ 4.1GB）与 Isaac 渲染不可共存，融合训练用
``vla_backend=vision_only``（冻结视觉塔 ~1.5GB bf16）作 in-sim actor。
"""

from __future__ import annotations

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
# v2：VLA-as-policy 融合（实现在 vla_policy.py / ppo_loop.py，这里重导出）
# ============================================================================
from .vla_policy import VLAAsPolicy  # noqa: E402

# train_vla_in_loop 在 ppo_loop.py（M2c 实现），此处惰性导入避免循环
def train_vla_in_loop(settings, **overrides) -> object:
    from .ppo_loop import train_vla_in_loop as _impl
    return _impl(settings, **overrides)
