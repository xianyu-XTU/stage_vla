"""sim_gateway.py —— 仿真网关抽象（模块②③ 与具体仿真解耦，M4 预留 Gazebo/ManiSkill）。

RL / 奖励 / 评估代码只依赖网关接口，不硬绑 Isaac Lab：

    SimGateway(ABC)
      ├── make_env(cfg)                    # 构造环境
      ├── step(action)                     # 执行动作
      ├── get_obs()                        # 读取观测
      ├── get_stage_signals()              # 读取阶段判定信号（末端/方块/抓取/堆叠）
      └── close()

M0 只实现 :class:`IsaacLabGateway`（薄包装 runner.create_env）；M4 增加
``GazeboGateway`` / ``ManiSkillGateway``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SimGateway(ABC):
    """仿真后端统一接口。"""

    @abstractmethod
    def make_env(self, cfg) -> None:
        """构造并启动环境。"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> None:
        """执行一步动作。"""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self):
        """返回当前观测。"""
        raise NotImplementedError

    @abstractmethod
    def get_stage_signals(self):
        """返回阶段判定所需信号（末端位置 / 方块位置 / 抓取与堆叠）。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        raise NotImplementedError


class IsaacLabGateway(SimGateway):
    """Isaac Lab 后端网关（**需 Isaac 环境**，惰性 import）。"""

    def make_env(self, cfg) -> None:
        raise NotImplementedError("IsaacLabGateway 为 M2 里程碑实现。当前直接使用 rl.runner.create_env。")

    def step(self, action) -> None:
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_stage_signals(self):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
