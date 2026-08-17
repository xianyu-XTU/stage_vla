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

## 2026-08-17 · success=0 结构性修复（按 E:\STAGE_VLA_SUCCESS_ZERO_FIX_HANDOFF.md）

**诊断结论**：success=0 不是训练量不够，而是结构性叠加。最重要的两个：
1. **success 定义错误**：任务 `Isaac-Stack-Cube-Franka-IK-Rel-v0` 的 `terminations.success`
   继承官方 `mdp.cubes_stacked`（**三块塔**：蓝压红、红压绿、夹爪开），而项目目标是
   `red-on-blue`（红压蓝）——**叠放方向相反，success 结构上不可能触发**。
2. **grasp 真值错误**：官方 `object_grasped` 是几何近似（末端<6cm+手指未全开），
   项目旧奖励 `finger<0.01` 奖励夹爪闭死——空手全闭骗分，真夹住方块反而闭不到 0.01。

**本轮修复（全部代码级落地，测试 72 项全绿）**：
- **success**：新增 `red_on_blue_success`（红块在蓝块上 + 夹爪释放 + 低速 + 持续 20 步），
  接线到 `cfg_surgery` / `train_stage` 的 `terminations.success`；`tools/diag_red_on_blue.py`
  做 scripted 冒烟验证（先验证再训练）。
- **真实抓取**：`stages/grasp.py` 纯几何判定（方块在两指之间 + 手指未全开 + 末端贴近）；
  弃用 `finger<0.01`；可选 ContactSensor（`--use_contact_sensor`，几何代理为默认）。
- **grasp-gated 奖励**：lift/move/stack 全部以 stable_grasp 门控；move 势能改 cube_2→cube_1
  （旧 EE→底座，空手走也能骗分）；新增 `drop_penalty` 掉块惩罚；
  修 `train_stage` 的 `progress_shaping` 重复乘权重（3×3=9 → 单次）。
- **帧级缓存**：reward manager 跳过 weight==0 的 term，`_grasp_state` 用
  `common_step_counter` 做**同帧幂等**缓存，避免"连续抓取计数器"同帧重复累加。
- **诊断指标**：`rl/diagnostics.StageMetrics` + `tools/collect_diagnostics.py` +
  `run_staged_pipeline.py` 输出 16 节指标表（physical_grasp / stable_grasp / lift /
  survival_after_lift / released_stack / task_success 等）。
- **stage reset 课程**：`rl/state_bank.py`（采集稳定抓取成功态 → 下游阶段 80% 采样重置，
  opt-in `--state_bank` / `--bank_from`）；`train_stage --init_ckpt`（策略继承渐进微调）；
  `run_staged_pipeline` 增加掉块回退。

**待 Isaac 环境验证**（本机已做纯逻辑单测，未跑真训练）：
- [ ] `tools/diag_red_on_blue.py` scripted 冒烟：确认 success 能触发
- [ ] 短训（grasp 200-500 iter）看 physical_grasp_rate > 0
- [ ] ContactSensor 路径 `--use_contact_sensor`（需先 `tools/diag_grasp` 验证接触力阈值）
- [ ] state_bank 课程（先 grasp 采库 → lift 从库重置）

## 2026-08-17 · 实验验证轮（success 冒烟 + 物理抓取定位）

**① success 冒烟验证通过**（`tools/diag_red_on_blue.py`）：
- scripted 精确放置（红块静止高度 0.0468 对齐到蓝块上、零速、夹爪开）→ 稳定 20 步 →
  `red_on_blue_success` 触发 ✅。success 定义修复正确。
- 关键物理修复：默认 PhysX TGS 下静态堆叠 ~10 env step 就散架（"noisy velocities" 警告），
  `enable_external_forces_every_iteration=True`（cfg_surgery 注入）让稳定窗口翻倍到 20 步。
  这解释了前一轮 success 恒 0 的一个隐藏根因——即使叠好，物理也留不住 20 步。

**② 抓取奖励问题定位**：
- 500 迭代 grasp 新策略：mean reward 高（~250）但 **physical_grasp_rate=0**——策略只拿
  `opp`（对侧定位，值≈1）的免费分，从不闭合（close≈0、physical≈0）。已把 `opp_scale`
  从 1.0 压到 0.3，让闭合/物理抓取成为主要收益。
- 对照：**旧 2000 迭代 grasp 策略（声称"稳定抓 330 步"）physical_grasp_rate 也是 0**——
  印证"330 步"是旧几何信号 `object_grasped`（末端<6cm+手指未全开）在 hover 时误报，
  从没有任何策略真正物理抓取过。

**③ 物理抓取几何定义修复（关键）**：
- 根因：`_physical_grasp` / `object_grasp_combined_reward` 用 `robot.data.body_pos_w`
  的 `panda_leftfinger/rightfinger`，那是手指**根/枢轴**（近手心），不是指尖 →
  "方块在两指之间"判定永远不成立。
- 修复：改用 FrameTransformer 的指尖帧 `ee_frame.data.target_pos_w[:, 1]`（右指尖）/
  `[:, 2]`（左指尖）（stack_ik_rel_env_cfg 里配的 tool_rightfinger/tool_leftfinger，
  含 0.046 偏移）。
- scripted 自适应下降验证：指尖降到方块中心高度（z≈0.047，方块顶 0.0437）后
  `between=True`，闭合后 **physical=True** ✅。定义可达。

**④ 剩余终极瓶颈（精确定位）**：即使 `physical=True`，抬升时方块不被带走——
脚本化抓取时指尖夹在方块**上缘**（z=0.047 略高于方块顶 0.0437），夹的是顶角，
一抬就滑脱（cube 留在桌面 z=0.020）。即前一轮"物理抓取终极瓶颈"= 抓取高度/指尖接触点
问题，不是 reward 学不会。策略需学到**中部稳定抓持**（指尖对准方块垂直中心）才能抬升。

## 2026-08-17 · grasp 重训实验（v3，全修复版）

**迭代记录**：
- v1（新奖励原版，500 iter）：reward ~250 但 **physical_grasp=0**——`opp` 免费分主导，策略 hover；
- v2（opp_scale=0.3，600 iter，被打断）：reward 降到 ~78，仍 hover（值≈0.30=opp×0.3）；
- v3（+height_align 引导下降，600 iter）：**reward 76→147**，**physical_grasp_rate 0→0.001**，
  策略开始"下降+闭合+物理抓取"（诊断：episode 1 提前 153 步终止=抓取后掉块）。

**结论**：
- 指尖帧修复 + opp 压权 + height_align 引导下降，三层共同让"物理抓取"从**不可达**变成
  **偶尔可达**（0.1% 步数）。reward 学出来的行为明显强于纯 hover。
- 剩余瓶颈：physical grasp 仍罕见（0.1%）、stable_grasp_10/20step 仍 0——策略抓到就掉。
  需要更长训练（旧会话 grasp 用 2000 iter）或 stable-grasp 专项奖励/课程。

**下一步候选**：① grasp 延长到 1500-2000 iter 看 stable_grasp 能否>0；② 用 state_bank 采集
稳定抓取态做 curriculum；③ 攻 lift 稳定（指尖在中部抓持，避免上缘滑脱）。

## 2026-08-17 · grasp 3000 迭代长训（v4，128 env）

**结果**：3000 迭代（128 env）mean reward 160→**214**（末值，区间 160-234），
grasp 奖励 value≈**0.88**（hover 基线 0.31），`tools/plot_curves.py` 已出曲线
（`outputs/training_curves.png`，含 stage_goal/stage_grasp_hold/drop_penalty）。

**但诊断暴露最后一个奖励漏洞**：`collect_diagnostics` 显示 **physical_grasp_rate=0.000**！
策略 value≈0.88 全靠 `close`（闭合×贴近，最高 3.0）+ `height_align`（1.5）——
学会了"下降+闭合空手"，但方块从没真正在两指之间（between=False → physical=0）。

**修复**：`object_grasp_combined_reward` 的 `close` 项用 `cube_between_fingers` 门控
（`close = between * closing * near`）——只有方块真的在两指之间，闭合才给分。这消除
"闭合空手骗分"，策略必须先把指尖放到环绕方块的位置再闭合。

## 待记录（后续里程碑）

- [ ] M2 更长视觉融合训练看阶段覆盖率
- [ ] M3 轻量化对比表
- [ ] M4 跨仿真泛化结果
