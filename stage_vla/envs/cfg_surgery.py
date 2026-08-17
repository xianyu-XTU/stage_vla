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
    use_contact_sensor: bool = False,
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

        # 项目自定 success（红块在蓝块上）替代官方 cubes_stacked 三块塔——
        # 官方判定与项目 red-on-blue 目标叠放方向相反，导致 success 恒 0
        from isaaclab.managers import TerminationTermCfg as DoneTerm

        from ..stages.rewards_isaac import red_on_blue_success

        env_cfg.terminations.success = DoneTerm(func=red_on_blue_success)

        # 物理求解器调优：静态堆叠稳定（diag 实测默认 ~10 步散架）
        _stabilize_stack_physics(env_cfg)

        if use_contact_sensor:
            _add_contact_sensors(env_cfg, cube_grasp)

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

    # 项目自定 success（同 build_stage_env_cfg，M2 视觉线也按真实任务目标评价）
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    from ..stages.rewards_isaac import red_on_blue_success

    env_cfg.terminations.success = DoneTerm(func=red_on_blue_success)

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


def _stabilize_stack_physics(env_cfg) -> None:
    """物理求解器调优：让**静态堆叠稳定**（对训练与 diag 同时生效）。

    diag 实测：默认 PhysX TGS 配置下，"红块完美叠在蓝块上"的静态堆叠只能维持
    ~10 env step 就散架（日志有 "TGS solver ... may cause noisy velocities" 警告），
    success 的 20 步持续判定无法触发。
    ``enable_external_forces_every_iteration`` 让每子步迭代外部力，改善静置接触
    数值稳定性。经验：改方块摩擦反而把"水平滑开"变成"垂直弹飞"（更糟），故不动材质。
    """
    if getattr(env_cfg.sim, "physics", None) is not None:
        env_cfg.sim.physics.enable_external_forces_every_iteration = True


def _add_contact_sensors(env_cfg, cube_to_grasp: str) -> None:
    """给 Franka 左右指注入 ContactSensor（过滤体为被夹方块），供物理抓取判定用。

    ``rewards_isaac._physical_grasp`` 会读取 ``scene.lf_contact / rf_contact`` 的
    ``force_matrix_w`` 叠加接触力条件（更接近物理真值，交接文档 5 节）。

    .. note::
        默认**不启用**（几何代理 ``physical_grasp_geometric`` 已可用）。启用后务必先用
        ``tools/diag_grasp.py`` 在 Isaac 环境验证一次接触力阈值，再开长训。
    """
    from isaaclab_physx.sensors import ContactSensorCfg

    cube_name = cube_to_grasp.split("_")[-1]              # "cube_2" -> "2"
    cube_prim = f"{{ENV_REGEX_NS}}/Cube_{cube_name}"      # prim 路径 {ENV_REGEX_NS}/Cube_2
    for name, finger in (("lf_contact", "panda_leftfinger"), ("rf_contact", "panda_rightfinger")):
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger}",
                filter_prim_paths_expr=[cube_prim],
                update_period=0.0,
            ),
        )
