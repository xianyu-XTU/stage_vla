# 架构总览

## 申请书三模块 ↔ 代码位置

| 申请书承诺 | 代码位置 | 文档 |
|---|---|---|
| 模块① 长程任务阶段感知机制（无监督分离器 / 阶段计算器 / 稠密奖励） | `stage_vla/stages/` | `docs/module1_stages.md` |
| 模块② StARe-RL 与 VLA 融合（OpenVLA + PPO → StARe-PPO / 动作接口 / 在线反馈） | `stage_vla/rl/` + `stage_vla/policies/` | `docs/module2_fusion.md` |
| 模块③ 模型轻量化（LoRA / INT8/INT4 / 蒸馏） | `stage_vla/lightweight/` | `docs/module3_lightweight.md` |
| 仿真（Gazebo / Isaac Sim）+ 数据集（ManiSkill 3 / SimpleReNV） | `stage_vla/envs/` + `stage_vla/data/` | `docs/simulation.md` |
| 预期成果（算法框架 / 源码 / 研发手册 / 实验报告） | 全部 + `docs/research_log.md` + `outputs/reports/` | `docs/roadmap.md` |

## 数据流

**线1（阶段感知 RL）**

```
语言指令
  → semantic.parse()  → SemanticPlan(子目标 + 阶段序列)
  → envs.cfg_surgery   → 注入阶段感知奖励（rewards_isaac.build_stage_rewards_cfg）
  → rl.runner.train    → OnPolicyRunner.learn
      每步：StageDetector 读几何信号 → calculator.signals_to_stage/progress
            → potential_shaping + stage_completion_reward（首帧置零，/dt 抵消）
  → rl.eval            → 阶段检测判 grasp/stack 成功率
```

**线2（轻量 VLA 部署，M2/M3）**

```
相机帧 + 指令
  → policies.build_policy(name)  → VLAPolicy.get_action
  → rl.action_interface.to_env_action → env.step
  → lightweight.{lora, quantize, distill} → 轻量化模型
```

## 设计原则（与旧工程差异）

1. **单一配置源**：只读 `config/default.yaml`，无平行 `_DEFAULTS` 常量（旧工程两处漂移）。
2. **机器无关**：提交内容零盘符路径，真实路径只进 gitignored 的 `config.local.yaml`。
3. **纯函数层**：阶段判定逻辑收敛在 `stages/calculator.py` 单一实现，detector / rewards /
   env 全部复用（防逻辑漂移）。
4. **惰性 import**：Isaac 依赖全部函数内 import，纯配置/无 Isaac 环境下可安全导入核心包。
5. **按名构造**：策略/任务经 `core.registry` 注册，脚本只调 factory（根治兄弟脚本互相 import）。
6. **INT8 不造假**：未接入真实 INT8 前直接抛错，绝不静默走 bf16。
7. **无嵌套 git**：第三方代码禁止 git clone 进仓库（见 `third_party/README.md`）。
