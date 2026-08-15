# 模块② StARe-RL 与 VLA 融合框架

## 目标（申请书承诺）

以 OpenVLA 为基础模型，融合 PPO 强化学习框架构建 StARe-PPO；优化 VLA 动作输出接口，
结合在线反馈机制，让模型能根据环境变化（物体位置偏移、碰撞、光照变化）实时调整动作。

## 架构

| 组件 | 文件 | 状态 |
|---|---|---|
| PPO 训练运行器（rsl_rl 5.x，v1 状态版） | `rl/runner.py` + `rl/cfg.py` | ✅ M1 |
| **VLA-as-policy 融合训练（自研 PPO 循环）** | `rl/ppo_loop.py` | ✅ M2（视觉融合 10 迭代跑通，显存 1.63GB） |
| VLAAsPolicy 接口 + VisionOnlyPPOPolicy | `rl/vla_policy.py` | ✅ M2（冻结视觉塔 + CPU 可训头） |
| 动作输出接口（VLA ↔ 环境动作） | `rl/action_interface.py` | ✅ M2（IKRel/JointPos + 128 维往返） |
| 阶段反馈回灌 | `rl/online_feedback.py` | ✅ M2（StageFeedback → one-hot 条件输入） |
| 视觉环境配置（id_visuomotor + 阶段奖励） | `envs/cfg_surgery.py` | ✅ M2 |
| 策略注册/按名构造 | `policies/factory.py` + `core/registry.py` | ✅ |

## 8GB 显存铁律

OpenVLA-7B（4-bit ≈ 4.1GB）与 Isaac 渲染**不可共存**。融合训练闭环方式：
- `rl.vla.loop_mode = in_sim`（仅 `vision_only` 后端可行）
- `record_replay`（默认）：先 record 再 replay，避免同时加载模型与渲染
- `tcp_separate`：deploy/server.py + client.py 双进程分离

## 验收（M2）

- 动作往返测试通过（to_env_action → env.step → from_env_state 一致）
- `train_stare.py --vla vision_only` 闭环训练跑通（8GB 内）
- 有无阶段反馈的 reward/成功曲线可区分
