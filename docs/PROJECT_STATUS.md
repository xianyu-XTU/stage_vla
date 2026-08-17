# stage_vla 项目现状总结（供外部咨询）

> 本文档由 Claude 整理，用于向其他 AI/专家咨询未解决的核心问题。自包含，无需额外背景。
> 最后更新：2026-08-17

---

## 1. 项目是什么

**湘潭大学大学生创新训练项目**：《基于阶段感知强化学习的轻量化 VLA 机械臂研究》

用 **视觉-语言-动作模型（VLA）+ 阶段感知强化学习（StARe-PPO）** 驱动 Franka 机械臂完成
"抓取方块→堆叠"长程任务，并对 VLA 模型轻量化（LoRA + 量化 + 蒸馏），目标消费级显卡
（RTX 3060/4060，8GB）可训练。

**核心创新点**：把长程任务拆成语义子阶段（approach→grasp→lift→move→stack），
给每阶段稠密奖励，解决 RL "奖励稀疏、学不动"的问题。

---

## 2. 硬件与仿真环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060（8.6GB 显存） |
| 仿真 | Isaac Lab 3.0.0 + 内嵌 Isaac Sim（`E:\work\IsaacLab`） |
| RL 库 | rsl_rl 5.0.1（PPO） |
| 任务 | `Isaac-Stack-Cube-Franka-IK-Rel-v0`（状态版）/ `-Visuomotor-v0`（视觉版）|
| 机器人 | Franka Panda（7 臂 + 2 指），IK 相对控制（7 维动作：6 末端增量 + 1 夹爪）|
| 方块 | 5cm 红色方块（cube_2，要抓），5cm 蓝色方块（cube_1，堆叠底座）|

**8GB 硬约束**：OpenVLA-7B 无法与渲染共存 → 用轻量/冻结模型；相机分辨率压到 96-128；
可训练头放 CPU（GPU backward 与 Isaac 渲染冲突会 device assert）。

---

## 3. 已完成的工作（全部可运行、有数据）

### 3.1 工程基础
- 私有 GitHub 仓库 `xianyu-XTU/stage_vla`，干净骨架、机器无关配置、CI 卫生检查
- 54 项单元测试全绿

### 3.2 阶段感知 RL（线1，状态版）
- **阶段检测器**：向量化几何判定（approach/grasp/lift/move/stack）
- **稠密奖励**：势能塑形 `γφ(s_t)−φ(s_{t−1})` + 阶段完成奖 + 动作惩罚
- **对比实验**（400 迭代）：阶段感知组 mean reward **29.3 vs 基线 -0.2**，阶段塑形有效
- **1000-3000 迭代长训**：reward 从 -0.3 涨到 51-88，**但完整任务 success 始终 0**

### 3.3 官方机制应用
- 照官方 `Isaac-Lift-Cube-Franka-v0` 加 **`object_is_lifted` 奖励**（方块被抬起高权重 15）
  → 策略开始抬起方块（训练中 lifting_object 峰值 14.7，视频验证抬起 390/400 步）
- **防奖励黑客**：阶段完成奖只发首次（修复反复跨阶段刷奖）

### 3.4 分阶段 RL（课程式，当前主线）
把任务拆成 4 个独立 RL 问题，各自训练：
| 阶段 | 奖励 | 单独效果 |
|---|---|---|
| **grasp** | 对侧抓取 + 显式闭合（社区方案） | ✅ 稳定抓住 330 步（隔离测试）|
| **lift** | 方块被抬起（权重 15） | ⚠️ 抬起信号触发但只 ~3% |
| **move** | 方块靠近目标 `1−tanh(d/std)` | ✅ 信号明显 |
| **stack** | 稠密堆叠（水平对齐×高度对齐） | ⚠️ 有信号但没稳定堆叠 |

每个阶段独立 checkpoint（`logs/stage_<name>/`），有组合执行脚本。

### 3.5 其他成果
- **VLA-light**：去 LLM 的轻量 VLA（指令分词器 29M 替代 Llama-2 7B，体积 -88%）
- **视觉引导的基础动作策略**：指令→基础动作分解 + 视觉→当前动作分类（40 张数据上 acc 1.0）
- **训练曲线工具**：`tools/plot_curves.py` 出 PNG
- **视频渲染**：`scripts/render_grasp_video.py` 出 mp4（抬起方块视频）

---

## 4. ⭐ 核心未解问题：完整任务 success 恒为 0

**现象**：所有训练（单策略 1000-3000 迭代、分阶段 RL）的完整任务成功率（`Episode_Termination/success`）**始终为 0**。
策略能学会"接近→抓取→抬起"（各阶段信号都触发），但**从未完成"移动→堆叠"的完整闭环**。

**分阶段组合执行**（串联 4 个策略）：到达最高阶段 = **grasp**，交接给 lift 时抓持丢失（方块掉落）。

---

## 5. 排查过的根因（已排除 vs 待解）

### 已排除
| 假设 | 结论 |
|---|---|
| 方块太大够不着 | ❌ 排除：Franka 手指全开 **80mm**（实测），> 方块 50mm |
| 末端高度/定位不对 | ❌ 排除：自适应再对齐到 0.2mm，指尖到方块中心高度，仍失败 |
| 下降撞走方块 | ❌ 排除：实测方块位移 0.0mm |
| 摩擦不够 | ❌ 排除：高摩擦（3.0）仍掉落 |

### 已定位（关键发现）
1. **`object_grasped` 是几何判定，不是物理判定**：只检查"方块距末端 <6cm 且手指闭合"，
   **不检查方块是否真在手指之间** → 信号触发≠真夹住。
2. **手指对侧问题**（社区 Issue #204 确认）：朴素距离奖励产生"手指同侧"局部最优。
   → **已用对侧抓取奖励缓解**（grasp 策略稳定 330 步），但交接时仍丢。
3. **物理抓取可靠性是终极瓶颈**：即使单个 grasp 策略能稳定抓住，
   **跨阶段交接**（grasp→lift）要求"稳定抓持"硬前提，而 lift 策略接手后握不住。

### 待解（核心疑问）
**如何让 RL 策略稳定地"抓稳并抬起来，然后不丢地完成移动和堆叠"？**
具体表现为：分阶段交接 grasp→lift 时，lift 策略（单独训练抬升）接手后丢抓取。

---

## 6. 已尝试的奖励方案与数据

| 方案 | 关键代码 | 效果 |
|---|---|---|
| 阶段势能塑形（非饱和距离势能） | `stage_vla/stages/rewards_isaac.py:_shaping_potential` | reward 学习，但 success 0 |
| 阶段完成奖（首次进入） | `rewards.stage_completion_reward_first_time` | 防黑客，但阶段学习变弱 |
| object_is_lifted（官方机制） | `rewards_isaac.object_is_lifted_reward` | 抬起信号出现（峰值14.7）|
| 对侧抓取奖励（社区 Issue #204） | `rewards_isaac.object_grasped_opposite_reward` | grasp 稳定 330 步 |
| 显式闭合奖励 | `rewards_isaac.object_grasp_combined_reward` | 补闭合动作 |
| 稠密堆叠（水平×高度对齐） | `rewards_isaac.object_stacked_dense_reward` | 有信号但未稳定 |

**训练数据**：
- 3000 迭代长训：mean_reward 0→51，lifting_object 峰值 14.7，success 0
- grasp 阶段：稳定抓住 330 步（隔离）
- 组合执行：到达 grasp 后交接丢失

**训练曲线**：`outputs/training_curves.png`，**视频**：`outputs/grasp_video.mp4`（抬起方块）。

---

## 7. 关键代码位置

| 组件 | 路径 |
|---|---|
| 阶段奖励 | `stage_vla/stages/rewards_isaac.py`、`rewards.py` |
| 阶段检测器 | `stage_vla/stages/detector.py`、`calculator.py` |
| 分阶段训练 | `scripts/train_stage.py` |
| 组合执行 | `scripts/run_staged_pipeline.py` |
| 视频渲染 | `scripts/render_grasp_video.py` |
| 训练曲线 | `tools/plot_curves.py` |
| 配置 | `config/default.yaml`（reward_weights / thresholds）|

---

## 8. 想请外部 AI 帮忙解答的具体问题

1. **如何让 RL 策略稳定完成"抓稳→抬起→移动→堆叠"？**
   - 分阶段交接 grasp→lift 丢抓取，是否有更好的交接/组合策略？
   - 是否应该让单个策略学完整序列（而非分阶段）？分阶段有什么正确做法？
2. **物理抓取在 Isaac Lab 中不稳的根本解决**：除了对侧奖励，还有什么机制能保证
   "手指在方块两侧且夹稳"？是否需要改物理参数（摩擦/接触/夹爪刚度）？
3. **分阶段 RL 的正确实现**：各阶段策略如何训练/交接才能串联成功？
   是否需要每阶段从上一阶段的结束状态初始化（而不是都从桌面初始）？
4. **是否应该用联合训练/课程学习/层级 RL**？还是直接加大训练量（官方 12000 迭代）？
5. **success 恒 0 是否意味着奖励设计或任务定义有问题**？如何诊断？

---

## 附：环境访问方式（如需复现）
- 项目根：`E:\stage_vla_v2`
- 运行训练：`python tools\run_isaaclab.py scripts\train_stage.py --stage grasp --max_iterations 2000`
- 配置：`config\default.yaml`
- 所有产物（曲线/视频/checkpoint）在 `outputs/` 和 `logs/`
