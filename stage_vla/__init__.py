"""stage_vla —— 基于阶段感知强化学习的轻量化 VLA 机械臂研究。

湘潭大学创新训练项目 · v2 全新骨架。

包结构对应申请书三大模块：

- ``stages``  模块① 长程任务阶段感知机制（语义分离 / 阶段检测 / 阶段计算器 / 稠密奖励）
- ``rl``      模块② StARe-RL 与 VLA 融合框架（PPO 训练 / 动作输出接口 / 在线反馈）
- ``lightweight`` 模块③ 模型轻量化（LoRA / INT8/INT4 量化 / 知识蒸馏）
- ``policies`` VLA 策略封装（OpenVLA / RDT / vision_only）
- ``envs``    仿真网关抽象（Isaac Lab 优先，预留 Gazebo / ManiSkill）
"""

__version__ = "0.1.0"
