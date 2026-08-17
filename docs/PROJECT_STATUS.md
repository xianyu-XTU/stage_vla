# stage_vla 项目现状总结（供外部咨询 · v2 更新）

> 本文档由 Claude 整理，用于向其他 AI/专家咨询未解决的核心问题。自包含，无需额外背景。
> 最后更新：2026-08-17（本版含上次咨询后(2026-08-17)的全部修复与实验数据）

---

## 1. 项目是什么

**湘潭大学大学生创新训练项目**：《基于阶段感知强化学习的轻量化 VLA 机械臂研究》

用 **视觉-语言-动作模型（VLA）+ 阶段感知强化学习（StARe-PPO）** 驱动 Franka 机械臂完成
"抓取方块→堆叠"长程任务，并对 VLA 模型轻量化（LoRA + 量化 + 蒸馏），目标消费级显卡
（RTX 4060，8GB 显存）。

**核心创新点**：把长程任务拆成语义子阶段（approach→grasp→lift→move→stack），
给每阶段稠密奖励，解决稀疏奖励"学不动"的问题。

---

## 2. 硬件与仿真环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060（8.6GB 显存） |
| 仿真 | Isaac Lab 3.0.0 + Isaac Sim 6.0.1 |
| RL 库 | rsl_rl 5.0.1（PPO，自研 VLA-PPO 循环另见 `rl/ppo_loop.py`） |
| 任务 | `Isaac-Stack-Cube-Franka-IK-Rel-v0`（状态版 IK 相对控制） |
| 机器人 | Franka Panda（7 臂 + 2 指），7 维动作（6 末端增量 + 1 夹爪，0=开/-1=闭） |
| 方块 | 5cm 红块（cube_2，要抓）+ 5cm 蓝块（cube_1，底座）+ 5cm 绿块（cube_3，干扰） |

---

## 3. ⭐ 上次咨询后的核心修复（2026-08-17，全部有实验数据）

上次咨询说"success 恒 0，物理抓取是终极瓶颈"。本轮逐一定位并修复：

### 3.1 【最重要】success 定义错误 —— 结构上不可能触发（已修 + 验证）
- 任务 `terminations.success` 继承官方 `mdp.cubes_stacked`：要求**三块塔**（蓝压红 + 红压绿 +
  夹爪开）。而项目目标是 **red-on-blue**（红压蓝）——**叠放方向相反**，即使红块完美放好，
  success 也**结构上永远为 0**。
- 修复：自定义 `red_on_blue_success`（红块在蓝块上 + 夹爪释放 + 低速 + 持续 20 帧）。
- **scripted 冒烟验证通过**（`tools/diag_red_on_blue.py`）：精确放置 → 稳定 20 帧 → success 触发 ✅。

### 3.2 物理求解器 —— 静态堆叠稳定性（已修 + 验证）
- 实测：默认 PhysX TGS 下，完美静态堆叠 ~10 env step 就散架（"noisy velocities" 警告）。
  这解释了为什么即使策略真的叠好，20 帧持续 success 也永远达不到。
- 修复：`enable_external_forces_every_iteration=True`（`cfg_surgery._stabilize_stack_physics`）
  → 稳定窗口 **10→20 帧**，恰好够 success 判定。

### 3.3 物理抓取几何定义 —— 指尖帧 bug（已修 + 验证）
- 根因：`_physical_grasp` 用 `body_pos_w[panda_leftfinger/rightfinger]`，那是手指**根/枢轴**
  （近手心）不是指尖 → "方块在两指之间"判定永远不成立。
- 修复：改用 FrameTransformer 的指尖帧（`ee_frame.data.target_pos_w[:,1]` 右/`[:,2]` 左）。
- **scripted 验证**（`tools/diag_physical_grasp.py`）：指尖降到方块高度+闭合 → `physical=True` ✅。

### 3.4 抓取奖励重平衡（三轮迭代）
| 版本 | 改动 | reward | physical_grasp |
|---|---|---|---|
| v1（500 iter） | 新奖励原版 | ~250（hover 虚高） | 0.000 |
| v2（opp 压到 0.3） | 去免费 hover 分 | ~78 | 0.000 |
| v3（+height_align 引导下降） | 分层引导下降 | 76→147 | **0.001** |
| v4（3000 iter, 128 env） | 长训 | 160→**214** | **0.000（见 3.5）** |

### 3.5 3000 迭代长训暴露的【最后一个奖励漏洞】（已修，未重训验证）
- 3000 迭代 grasp 策略 reward 学到 214、grasp 奖励 value≈0.88，但 **physical_grasp_rate=0.000**。
- 根因：`close` 奖励项（手指闭合 × 贴近，权重 3.0）**不要求方块在两指之间**——
  策略学会"下降 + 闭合空手"骗分（value 0.88 全靠 close+height_align，physical 为 0）。
- 修复：`close` 项用 `cube_between_fingers` 门控（`close = between * closing * near`），
  只有方块真的在两指之间，闭合才给分。**尚未重训验证**。

---

## 4. 当前核心未解问题（本次想请外部 AI 解答）

**问题收敛为一条：如何让 grasp 策略真正学到"稳定物理抓取 + 抬升不掉"？**

具体表现：
- 3000 迭代 grasp 只训出"接近/下降/空手闭合"（reward 214 但 physical_grasp=0）；
- 即使 scripted 抓取 `physical=True`（指尖环绕方块+闭合），**抬升时方块滑脱**（cube 留桌面）——
  脚本化抓取时指尖夹在方块**上缘**（z≈0.047 vs 方块顶 0.0437），夹的是顶角不是中部；
- 因此完整任务 success 仍为 0（grasp 阶段都未稳定，更未训 lift/move/stack）。

### 具体问题
1. **奖励设计**：除已修的 close 门控，还有哪些抓取奖励的坑？如何让策略学到"指尖先环绕方块
   中部、再闭合"的稳定抓持（而不是空手闭合/夹上缘）？是否需要把"抓持高度"显式编码进奖励？
2. **抬升稳定**：Franka 夹 5cm 方块、scripted 抓上缘就滑脱——是抓取高度/接触点问题，
   还是需要改物理（摩擦/接触模型/夹爪刚度）？有什么办法让抓持落在方块中部？
3. **课程/结构**：分阶段独立训练 + state-bank 重置（已实现，`--state_bank`）是不是正确路径？
   还是应该 grasp 阶段直接并入 lift（单个策略学"抓+抬"）避免交接？
4. **真实抓取真值**：已接好 ContactSensor 接口（`--use_contact_sensor`）但未验证。几何代理
   （两指间+未全开+贴近）是否够，还是必须用接触力？
5. **物理可行性**：RTX 4060 8GB + Isaac Lab TGS + 5cm 方块，稳定抓取是否本质上很难？
   有没有已知的 Franka+StackCube 稳定抓取配置（摩擦/刚度/抓取位姿）？

---

## 5. 关键代码位置

| 组件 | 路径 |
|---|---|
| 项目自定 success | `stage_vla/stages/rewards_isaac.py:red_on_blue_success` |
| 物理抓取（指尖帧） | `stage_vla/stages/rewards_isaac.py:_physical_grasp` |
| 纯几何抓取判定 | `stage_vla/stages/grasp.py` |
| 抓取奖励（opp/height/close/physical） | `rewards_isaac.object_grasp_combined_reward` |
| 阶段奖励/势能 | `rewards_isaac.py`、`rewards.py` |
| 物理求解器修复 | `envs/cfg_surgery.py:_stabilize_stack_physics` |
| 分阶段训练 | `scripts/train_stage.py`（`--stage grasp --max_iterations 3000`） |
| 阶段诊断指标 | `tools/collect_diagnostics.py` + `rl/diagnostics.py` |
| 冒烟/scripted 验证 | `tools/diag_red_on_blue.py`、`tools/diag_physical_grasp.py` |
| 训练曲线 | `tools/plot_curves.py` → `outputs/training_curves.png` |
| 研发记录 | `docs/research_log.md`（全部实验迭代） |

---

## 6. 环境访问方式（如需复现）
- 项目根：`E:\stage_vla_v2`（GitHub public：`https://github.com/xianyu-XTU/stage_vla`）
- 跑训练：`python tools\run_isaaclab.py scripts\train_stage.py --stage grasp --max_iterations 3000`
- 诊断：`python tools\run_isaaclab.py tools\collect_diagnostics.py --checkpoint logs\stage_grasp\model_2999.pt`
- 冒烟：`python tools\run_isaaclab.py tools\diag_red_on_blue.py`
- 配置：`config\default.yaml`；产物在 `outputs/`、`logs/`
