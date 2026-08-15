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
    stage_rewards: bool = True,
):
    """构造带阶段感知奖励的 Isaac 环境配置（**需 Isaac 环境**）。

    Args:
        settings: 解析后的配置（:class:`~stage_vla.core.config.Settings`）
        num_envs: 并行环境数
        seed: 随机种子
        task_id: 任务 id（缺省取 ``settings.task["id_state"]``）
        cube_to_grasp / cube_to_stack_on: 目标方块 / 底座方块（来自语义计划，缺省取 config）
        active_stages: 指令覆盖的活动阶段（语义计划驱动；未覆盖阶段完成奖置 0）
        stage_rewards: 是否注入阶段感知奖励（False = 保留任务默认奖励，作对照基线）

    Returns:
        Isaac Lab ``ManagerBasedRLEnvCfg``（默认注入阶段感知奖励）
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_id = task_id or settings.task["id_state"]
    # Isaac Lab 3.0：load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    if stage_rewards:
        from ..stages.rewards_isaac import build_stage_rewards_cfg
        from ..stages.semantic import filter_stage_weights

        cube_grasp = cube_to_grasp or settings.task["cube_to_grasp"]
        cube_stack = cube_to_stack_on or settings.task["cube_to_stack_on"]

        # 语义计划驱动：未覆盖的阶段完成奖置 0（不奖励指令没要求的子阶段）
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

    if stage_rewards:
        logger.info(
            "环境配置已注入阶段感知奖励：task=%s num_envs=%d seed=%d 目标=(%s→%s) 活动阶段=%s",
            task_id, num_envs, seed, cube_grasp, cube_stack, active_stages,
        )
    else:
        logger.info("环境配置使用任务默认奖励（基线）：task=%s num_envs=%d seed=%d", task_id, num_envs, seed)
    return env_cfg


def build_vision_env_cfg(
    settings,
    num_envs: int,
    seed: int,
    *,
    cube_to_grasp: str | None = None,
    cube_to_stack_on: str | None = None,
    active_stages: list[str] | None = None,
    cam_res: int = 128,
):
    """构造带相机 + 阶段感知奖励的**视觉**环境配置（M2 VLA 融合用，**需 Isaac 环境**）。

    用 ``id_visuomotor`` 任务（双相机 200×200 RGB + 7 维 IK-Rel 动作）：
    - 相机降分辨率到 ``cam_res``（8GB 显存省渲染）；
    - 注入阶段感知奖励；
    - **保持嵌套观测**（图像不能拼进状态向量，不设 ``concatenate_terms``）。
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from ..stages.rewards_isaac import build_stage_rewards_cfg
    from ..stages.semantic import filter_stage_weights

    task_id = settings.task["id_visuomotor"]
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    cube_grasp = cube_to_grasp or settings.task["cube_to_grasp"]
    cube_stack = cube_to_stack_on or settings.task["cube_to_stack_on"]
    weights = filter_stage_weights(settings.reward_weights, settings.stages, active_stages)
    env_cfg.rewards = build_stage_rewards_cfg(
        weights=weights,
        thresholds=settings.thresholds,
        stages=settings.stages,
        cube_to_grasp=cube_grasp,
        cube_to_stack_on=cube_stack,
        active_stages=active_stages,
    )

    # 相机降分辨率（省显存；不做 concatenate，图像观测保持 dict）
    for cam in ("table_cam", "wrist_cam"):
        if hasattr(env_cfg.scene, cam):
            setattr(getattr(env_cfg.scene, cam), "height", cam_res)
            setattr(getattr(env_cfg.scene, cam), "width", cam_res)

    logger.info(
        "视觉环境配置就绪：task=%s num_envs=%d seed=%d cam_res=%d 目标=(%s→%s) 活动阶段=%s",
        task_id, num_envs, seed, cam_res, cube_grasp, cube_stack, active_stages,
    )
    return env_cfg
