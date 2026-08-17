"""diagnostics.py —— 阶段诊断指标收集（交接文档 16 节）。

回答"reward 高但 success=0 到底死在哪一阶段"：不再只记一个 task success，
而是按 episode 记录分阶段信号，汇总成诊断指标表（``tools/collect_diagnostics.py``
与 ``scripts/run_staged_pipeline.py`` 共用）。

每步由调用方喂入（从 Isaac 环境读取的信号，本模块纯张量统计、无仿真依赖）：
    stage        当前阶段索引 [N]（0=approach 1=grasp 2=lift 3=move 4=stack）
    physical     物理抓取掩码 [N]（rewards_isaac._physical_grasp）
    cube_z       被夹方块高度 [N]
    cube_vel     被夹方块速度 [N,3]
    released     夹爪已释放 [N]（两指 isclose 全开）
    rb_frame     red_on_blue 单帧（对齐+释放+低速）[N]

输出指标（交接文档 16 节子集）：
    physical_grasp_rate / stable_grasp_{10,20}step_rate
    lift_5cm_rate / grasp_survival_after_lift_{10,20}step
    move_with_grasp_rate / red_on_blue_geometry_rate / release_rate
    stable_stack_{10,20}step_rate / task_success_rate
"""

from __future__ import annotations

import torch

# 指标阈值（与 config/代码默认对齐）
LIFT_HEIGHT = 0.05        # 判定"抬起了 5cm"的方块离桌高度
TABLE_Z = 0.02


class StageMetrics:
    """按 episode 收集阶段诊断指标（单环境 num_envs=1 使用）。"""

    def __init__(self, stable_steps: int = 20):
        self.stable_steps = stable_steps
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._steps = 0
        self._grasp_streak = 0
        self._max_grasp_streak = 0
        self._physical_steps = 0
        self._lift_first_step: int | None = None      # 首次进入 lift 的步号
        self._lift_5cm = False
        self._survive_10: bool | None = None
        self._survive_20: bool | None = None
        self._move_steps = 0
        self._move_grasp_steps = 0
        self._rb_streak = 0
        self._max_rb_streak = 0
        self._rb_geometry_steps = 0
        self._released_steps = 0

    def reset(self) -> None:
        """开始新 episode。"""
        self._reset_episode()

    def record(
        self,
        stage: int,
        physical: bool,
        cube_z: float,
        cube_vel: torch.Tensor,
        released: bool,
        rb_frame: bool,
    ) -> None:
        """记录一步。"""
        self._steps += 1
        if physical:
            self._physical_steps += 1
            self._grasp_streak += 1
            self._max_grasp_streak = max(self._max_grasp_streak, self._grasp_streak)
        else:
            self._grasp_streak = 0

        if cube_z - TABLE_Z >= LIFT_HEIGHT:
            self._lift_5cm = True

        if self._lift_first_step is None and stage >= 2:      # stage 2 = lift
            self._lift_first_step = self._steps
        if self._lift_first_step is not None:
            age = self._steps - self._lift_first_step
            if age == 10 and self._survive_10 is None:
                self._survive_10 = physical
            if age == 20 and self._survive_20 is None:
                self._survive_20 = physical

        if stage == 3:                                        # move
            self._move_steps += 1
            if physical:
                self._move_grasp_steps += 1

        if rb_frame:
            self._rb_streak += 1
            self._max_rb_streak = max(self._max_rb_streak, self._rb_streak)
            self._rb_geometry_steps += 1
        else:
            self._rb_streak = 0
        if released:
            self._released_steps += 1

    def end_episode(self, success: bool) -> None:
        """结束当前 episode，把逐 episode 布尔并入累计。"""
        ep = {
            "episodes": 1,
            "steps": self._steps,
            "physical_steps": self._physical_steps,
            "stable_10": 1 if self._max_grasp_streak >= 10 else 0,
            "stable_20": 1 if self._max_grasp_streak >= self.stable_steps else 0,
            "lift_5cm": 1 if self._lift_5cm else 0,
            "survive_10": 1 if self._survive_10 is True else 0,
            "survive_20": 1 if self._survive_20 is True else 0,
            "move_steps": self._move_steps,
            "move_grasp_steps": self._move_grasp_steps,
            "rb_geometry_steps": self._rb_geometry_steps,
            "released_steps": self._released_steps,
            "stack_stable_10": 1 if self._max_rb_streak >= 10 else 0,
            "stack_stable_20": 1 if self._max_rb_streak >= self.stable_steps else 0,
            "success": 1 if success else 0,
        }
        for k, v in ep.items():
            setattr(self, f"_acc_{k}", getattr(self, f"_acc_{k}", 0) + v)
        self._reset_episode()

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """汇总全部 episode 的诊断指标表。"""
        eps = max(getattr(self, "_acc_episodes", 0), 1)
        steps = max(getattr(self, "_acc_steps", 0), 1)
        move_denom = max(getattr(self, "_acc_move_steps", 0), 1)
        return {
            "episodes": getattr(self, "_acc_episodes", 0),
            "steps": getattr(self, "_acc_steps", 0),
            "physical_grasp_rate": getattr(self, "_acc_physical_steps", 0) / steps,
            "stable_grasp_10step_rate": getattr(self, "_acc_stable_10", 0) / eps,
            f"stable_grasp_{self.stable_steps}step_rate": getattr(self, "_acc_stable_20", 0) / eps,
            "lift_5cm_rate": getattr(self, "_acc_lift_5cm", 0) / eps,
            "grasp_survival_after_lift_10step": getattr(self, "_acc_survive_10", 0) / eps,
            "grasp_survival_after_lift_20step": getattr(self, "_acc_survive_20", 0) / eps,
            "move_with_grasp_rate": getattr(self, "_acc_move_grasp_steps", 0) / move_denom,
            "red_on_blue_geometry_rate": getattr(self, "_acc_rb_geometry_steps", 0) / steps,
            "release_rate": getattr(self, "_acc_released_steps", 0) / steps,
            "stable_stack_10step_rate": getattr(self, "_acc_stack_stable_10", 0) / eps,
            "stable_stack_20step_rate": getattr(self, "_acc_stack_stable_20", 0) / eps,
            "task_success_rate": getattr(self, "_acc_success", 0) / eps,
        }

    def print_report(self) -> None:
        m = self.report()
        print("=" * 64)
        print(f"阶段诊断指标（{m['episodes']} episodes / {m['steps']} steps）")
        print("=" * 64)
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<38} {v:6.3f}")
            else:
                print(f"  {k:<38} {v}")
