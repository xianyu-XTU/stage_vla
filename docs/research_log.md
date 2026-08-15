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

## 待记录（后续里程碑）

- [ ] M1 无监督阶段分离器方法对比
- [ ] M2 融合训练实验数据
- [ ] M3 轻量化对比表
- [ ] M4 跨仿真泛化结果
