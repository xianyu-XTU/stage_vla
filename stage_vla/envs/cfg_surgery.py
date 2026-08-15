"""cfg_surgery.py —— 环境配置手术（**需 Isaac 环境**，惰性 import）。

不改 Isaac Lab 源码，通过 ``load_cfg_from_registry`` 取已注册任务 cfg，然后：
- 覆写 ``num_envs / seed``；
- 注入阶段感知奖励（``rewards_isaac.build_stage_rewards_cfg``）；
- 防御性设 ``observations.policy.concatenate_terms = True``（rsl_rl 5.x 观测组要求）；
- 可选注入俯瞰相机（8GB 显存用低分辨率）。
"""

from __future__ import annotations

from ..core.logging import get_logger

logger = get_logger(__name__)


def build_stage_env_cfg(
    settings,
    num_envs: int,
    seed: int,
    task_id: str | None = None,
    *,
    cube_to_grasp: str | None = None,
    cube_to_stack_on: str | None = None,
    active_stages: list[str] | None = None,
):
    """构造带阶段感知奖励的 Isaac 环境配置（**需 Isaac 环境**）。

    Args:
        settings: 解析后的配置（:class:`~stage_vla.core.config.Settings`）
        num_envs: 并行环境数
        seed: 随机种子
        task_id: 任务 id（缺省取 ``settings.task["id_state"]``）
        cube_to_grasp / cube_to_stack_on: 目标方块 / 底座方块（来自语义计划，缺省取 config）
        active_stages: 指令覆盖的活动阶段（语义计划驱动；未覆盖阶段完成奖置 0）

    Returns:
        Isaac Lab ``ManagerBasedRLEnvCfg``（已注入阶段感知奖励）
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from ..stages.rewards_isaac import build_stage_rewards_cfg

    task_id = task_id or settings.task["id_state"]
    # Isaac Lab 3.0：load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    cube_grasp = cube_to_grasp or settings.task["cube_to_grasp"]
    cube_stack = cube_to_stack_on or settings.task["cube_to_stack_on"]

    # 语义计划驱动：未覆盖的阶段完成奖置 0（不奖励指令没要求的子阶段）
    from ..stages.semantic import filter_stage_weights

    weights = filter_stage_weights(settings.reward_weights, settings.stages, active_stages)

    # 注入阶段感知稠密奖励（权重/阈值/阶段/目标方块/活动阶段全来自 config+计划）
    env_cfg.rewards = build_stage_rewards_cfg(
        weights=weights,
        thresholds=settings.thresholds,
        stages=settings.stages,
        cube_to_grasp=cube_grasp,
        cube_to_stack_on=cube_stack,
        active_stages=active_stages,
    )

    # rsl_rl 5.x 观测组要求观测项合并为一个张量
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = True

    logger.info(
        "环境配置已注入阶段感知奖励：task=%s num_envs=%d seed=%d 目标=(%s→%s) 活动阶段=%s",
        task_id, num_envs, seed, cube_grasp, cube_stack, active_stages,
    )
    return env_cfg
