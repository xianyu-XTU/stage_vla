# 里程碑路线图

## M0 骨架 + 第一版可跑原型（本次交付 ✅）

- [x] 目录树 / 仓库卫生文件（.gitignore / .gitattributes / LICENSE / pyproject.toml）
- [x] 配置体系：`default.yaml`（机器无关）+ `config.local.yaml` + 三层合并加载器
- [x] 阶段感知核心（纯张量）：semantic / calculator / detector / rewards
- [x] 全链路预留接口：action_interface / online_feedback / VLAAsPolicy / distill / quantize
- [x] 第一版可跑原型：`scripts/smoke_test.py`（无需 Isaac）+ pytest
- [x] 工具：check_machine / check_repo_hygiene / run_isaaclab

**验收**：smoke_test exit 0 · pytest 全绿 · 无机器绝对路径 · 仓库已推送

## M1 模块① 完整

- [x] 无监督阶段分离器**真实实现**（`stages/unsupervised.py`：几何特征提取 +
      多维过分割(K×2) + 任务进度代理排序 + 贪心合并，脱离人工标注；合成轨迹一致性 >0.7）
- [x] 语义分离真正驱动训练（`semantic.plan_targets` 提取目标方块 + 活动阶段；
      `filter_stage_weights` 未覆盖阶段完成奖置 0；`train_stare` 解析指令驱动
      env cfg——已验证 "pick up only" → 活动阶段 [approach,grasp,lift]）
- [ ] `train_stare.py` 跑通，TensorBoard 出现五阶段覆盖率；与官方基线对比阶段塑形组累积奖励更高

**验收**：合成轨迹上聚类中心与几何 detect 一致性 > 阈值（✅ 已达成）；训练 reward 有限（✅ 已达成）

## M2 模块② 融合

- [ ] `action_interface` 三实现（IK-Rel / JointPos / VLA）往返测试
- [ ] `VLAAsPolicy` 进 PPO 闭环训练（8GB 默认 `vision_only` 或 record/replay / TCP 分离）
- [ ] `StageFeedback` 把阶段信号回灌 VLA 上下文
- [ ] `deploy/server.py` + `client.py` 双进程闭环

**验收**：`train_stare.py --vla vision_only` 闭环训练跑通（8GB 内）；有无阶段反馈曲线可分

## M3 模块③ 轻量化

- [ ] 演示链补齐：`collect_demos.py` → `outputs/demos/*.npz` → `finetune_lora.py`
- [ ] 真 INT8（optimum GPTQ/AWQ 或 torch.ao）+ INT4 运行时量化
- [ ] 蒸馏：OpenVLA-7B teacher → vision_only / LoRA-INT8 student
- [ ] `metrics.py` 对比参数量 / 显存 / 延迟

**验收**：`quantize_model.py --bits 8` 报告真实 INT8（非 bf16 静默回退）；grasp 成功率提升

## M4 仿真验证 + 结题

- [ ] `sim_gateway` 增加 ManiSkill 3 / SimpleReNV / Gazebo 网关
- [ ] 跨仿真泛化测试（同一权重在 Isaac Lab 与 ManiSkill 3 评估）
- [ ] 研发手册（`docs/research_log.md`）+ 结题实验报告（`outputs/reports/`）
