# 研发手册（持续记录）

记录关键决策、踩坑经验与实验结果，作为项目结题"标准化技术成果"的一部分。

## 2026-08-15 · 项目重建（v2 骨架）

- **决策**：全新骨架，不搬运旧 `E:\stage_vla` 代码（仅作参考）。
- **架构变更**：
  - 单一配置源 `config/default.yaml`，去除平行 `_DEFAULTS`（旧工程两处漂移根因）。
  - 阶段判定收敛到 `stages/calculator.py` 纯函数，detector/rewards/env 复用。
  - 提交内容零机器绝对路径；真实路径进 gitignored 的 `config.local.yaml`。
  - Isaac 依赖全部惰性 import，核心包可在无 Isaac 环境导入/单测。
  - INT8 不造假（未接入真实实现前抛错）。
- **已知坑（从旧代码吸收）**：
  - 奖励 `*dt` 抵消：返回 `值 / env.step_dt`。
  - reset 首帧：`episode_length_buf == 1` 识别，置零塑形/完成奖。
  - rsl_rl 5.x：`obs_groups` + `RslRlMLPModelCfg` + `handle_deprecated_rsl_rl_cfg`。
  - OpenVLA-7B（4-bit ≈4.1GB）与渲染不可共存 → record/replay / TCP 分离。

## M1 对比实验结果（2026-08-15，200 迭代 / 64 环境 / seed 42）

**阶段感知组 vs 基线（同 v2 管线、同 PPO 配置，仅奖励不同）**

| 指标 | 阶段感知（调优后） | 基线（任务默认奖励） |
|---|---|---|
| Mean episode return | **29.30** | -0.20 |
| 阶段跨越（stage_transition） | 0.54 | 0（从未抓到） |
| Episode length | ~450 | 600（跑满超时） |
| success | 0 | 0 |

**关键教训（奖励调优前后对比）**：
- 初版势能塑形 `φ=Σ进度列`（approach 列 `clamp(1-d/0.15)` 饱和）→ 随机游走净负偏置，
  200 迭代 mean reward 被拉到 **-2.68**，agent 学不动；
- 改为**非饱和距离势能** `φ=-(d_app+d_stack)` + `init_noise_std 1.0→0.3` →
  mean reward 从 0.03 单调爬升到 **29.30**，阶段跨越大量触发。
- 结论：阶段感知稠密奖励确实能引导稀疏任务的学习（相对稀疏基线），前提是势能设计要避免饱和偏置。

## M2 VLA 融合关键排障（2026-08-15）

**现象**：视觉融合训练（冻结视觉塔 + 可训头）在第一次 PPO 更新时
`CUDA error 710: device-side assert`（PhysX GpuArticulationView / RTX GpuCompute / Warp）。

**排查链**：零动作 step 稳定 → 视觉前向 + 动作 step 24 步稳定 → **GPU backward 是唯一触发点**。

**根因**：torch 的 **GPU backward 与 Isaac RTX 渲染 / Warp 物理同卡并发**触发 device assert；
forward（视觉塔、env.step）都稳定，唯独 autograd 反向传播冲突。（v1 无相机 + 小 MLP 未撞上。）

**修复（三件套）**：
1. **可训头常驻 CPU**（4.2M 参数足够快）：视觉塔 GPU 前向 → 特征 `detach().cpu()` → 头在 CPU 前向/反向/优化器。彻底避开 GPU backward 与渲染冲突。
2. **特征预提取**：每次 rollout 只跑一次视觉塔前向，特征存 buffer；PPO 多轮 minibatch 更新只回传头部。
3. **目标 detach**：收集时存的 `log_prob`/`value` 必须 `detach()`，否则更新时"backward through graph a second time"。

**附带修复**：rollout 展平按角色区分（特征/动作 [T,B,F]→[T*B,F]，标量 [T,B]→[T*B]），
否则 x 与 old_log_prob 维度不一致（IndexError）。

**结果**：`train_vla.py --num_envs 8 --max_iterations 10 --cam_res 96` 端到端跑通，
显存 1.63GB（预算内），PPO loss 正常更新。

## 待记录（后续里程碑）

- [ ] M2 更长视觉融合训练看阶段覆盖率
- [ ] M3 轻量化对比表
- [ ] M4 跨仿真泛化结果
