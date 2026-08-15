# stage_vla —— 基于阶段感知强化学习的轻量化 VLA 机械臂研究

湘潭大学大学生创新训练项目 · v2 全新骨架（GitHub 私有仓库基座）

用 **视觉-语言-动作模型（VLA）** 驱动机械臂完成"抓取方块 → 堆叠"长程任务。核心创新是
**阶段感知强化学习（StARe-PPO）**：把长程任务拆成语义子阶段，给每个阶段稠密奖励，
解决传统 RL"奖励稀疏、学不动"的问题；并对 VLA 模型做**轻量化**（LoRA + INT8/INT4 +
知识蒸馏），目标消费级显卡（RTX 3060/4060）可训练。

```
语言指令 → 语义分离(子目标/阶段) → 阶段检测 → 阶段稠密奖励 → StARe-PPO 训练 → 评估
```

## 两条技术线（对齐申请书三模块）

| 线 | 申请书模块 | 内容 | 状态 |
|---|---|---|---|
| 线1 阶段感知 RL | 模块① 阶段感知机制 + 模块② StARe-PPO | 状态版 StARe-PPO（无相机）训练"抓取-堆叠" | v1 已设计（M1 落地训练） |
| 线2 轻量 VLA | 模块② VLA 融合 + 模块③ 轻量化 | OpenVLA/RDT 推理 + QLoRA/INT8/INT4/蒸馏 | 接口已预留（M2/M3 落地） |

## 目录结构

```
stage_vla_v2/
├── config/
│   ├── default.yaml               # 唯一配置源（机器无关，paths 为占位符）
│   └── config.local.yaml.example  # 本地覆盖模板（复制为 config.local.yaml）
├── stage_vla/                     # 主包（与仓库同名）
│   ├── core/                      # 配置加载 / 日志 / 异常 / 注册表
│   ├── stages/                    # 模块①：semantic / calculator / detector / rewards
│   ├── rl/                        # 模块②：runner / stare_ppo / action_interface / online_feedback
│   ├── policies/                  # VLA 策略：base / factory / openvla / rdt / vision_only
│   ├── lightweight/               # 模块③：lora / quantize / distill / metrics
│   ├── envs/                      # 仿真接入：cfg_surgery / sim_gateway
│   ├── data/                      # 演示采集 / ManiSkill / SimpleReNV（预留）
│   └── deploy/                    # VLA TCP 服务 / 遥操（预留）
├── scripts/                       # 入口脚本（薄壳）
├── tests/                         # pytest
├── tools/                         # 机器探测 / 卫生检查 / isaaclab 运行器
└── docs/                          # 架构 / 模块 / 里程碑 / 研发手册
```

## 快速开始

### 1. 环境准备

运行时用 **Isaac Sim kit python**（含 torch，无独立 .venv）：

```bat
:: 首次安装 pytest
"<isaac_sim>\kit\python\python.exe" -m pip install pytest
```

### 2. 配置本机路径

```bat
copy config\config.local.yaml.example config.local.yaml
:: 编辑 config.local.yaml 填入本机路径（isaaclab / isaac_sim / openvla / rdt）
```

### 3. 第一版可跑原型（无需 Isaac Sim）

```bat
"<isaac_sim>\kit\python\python.exe" scripts\smoke_test.py
"<isaac_sim>\kit\python\python.exe" -m pytest tests -v
"<isaac_sim>\kit\python\python.exe" tools\check_machine.py
```

### 4. Isaac 环境冒烟（需 Isaac Lab）

```bat
python tools\run_isaaclab.py scripts\check_env.py --num_envs 4 --max_steps 5 --headless
```

## 配置说明

- 所有配置集中在 `config/default.yaml`，改配置无需改代码。
- 真实路径只写在 gitignored 的 `config.local.yaml`（提交内容**零机器绝对路径**）。
- 详见 `config/README.md`。

## 环境依赖（只读引用，不复制）

以下为示例路径，真实值在 `config.local.yaml`：

- Isaac Lab 3.0.0：`E:\work\IsaacLab`
- Isaac Sim 独立版：`E:\isaac-sim-standalone-6.0.1-windows-x86_64 (2)`
- OpenVLA-7B：`D:\openvla\models\openvla-7b`
- RDT-1B / SigLIP / T5-XXL：`D:\vla_models\*`

## 里程碑

见 `docs/roadmap.md`：M0 骨架（当前）→ M1 模块①完整 → M2 模块②融合 → M3 模块③轻量化 → M4 仿真验证+结题。

## License

MIT（见 `LICENSE`）。第三方代码引入规范见 `third_party/README.md`。
