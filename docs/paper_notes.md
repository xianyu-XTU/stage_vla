# 论文学习笔记（19 篇参考文献）

> 来源：`E:\创新项目论文\` 19 篇 PDF（对应申请书参考文献 [1]–[19]，缺 [15] Gazebo）。
> 精读日期：2026-08-15。分组精读 + 结构化整理，映射项目三大模块。

## 总览：文献 ↔ 项目模块

| 文献 | 主题 | 项目模块 | 关键价值 |
|---|---|---|---|
| [1] RT-2 | VLA 开山作 | 模块② | 动作 token 化 + co-fine-tuning 范本（闭源巨型，仅参照） |
| [2] PaLM-E | 具身多模态 LM | 模块② | 多模态 token 注入 / 多任务正迁移（562B，仅参照） |
| [3] **OpenVLA** | 开源 VLA | 模块② | **项目基础模型首选**（开源可改、LoRA/int4 可轻量微调） |
| [4] HiTS | 分层 RL 时间子目标 | 模块① | "什么+何时"子目标、hindsight relabeling、testing |
| [5] **STARE-VLA** | 阶段感知强化学习 | 模块①② | **最接近本项目的相关工作**（阶段拆分+阶段级稠密奖励+PPO） |
| [6] LoRA | 低秩微调 | 模块③ | 消费级可训练的核心杠杆（只训 ~5% 参数） |
| [7] 轻量化综述 | 剪枝/量化/蒸馏 | 模块③ | 三级叠加可行性 + 结构化优先/PTQ 后需 fine-tune |
| [8] 知识蒸馏 | teacher-student | 模块③ | 样本效率 + 软目标公式（T、1/T² 缩放） |
| [9] PPO | 近端策略优化 | 模块② | RL 算法底座（clip 目标 + GAE） |
| [10] Transformer | 基础架构 | 模块② | VLA 主干（缩放点积注意力、多头） |
| [11] Generalist(RoboVLMs) | VLA 设计指南 | 模块② | 骨干选型结论（PaliGemma/KosMos 强、域内数据是关键） |
| [12] Dex-Net 4.0 | 双手抓取 | 模块① | 合成数据+域随机化 sim-to-real、统一奖励 |
| [13] TinyVLA | 轻量 VLA | 模块③ | <1B 骨干 + 扩散头，5% 可训练参数，快 20× |
| [14] AnoleVLA | 轻量 VLA(SSM) | 模块③ | Mamba 线性复杂度、消费卡可完整训练 |
| [16] Isaac Lab | 仿真框架 | 仿真 | GPU 并行 RL 训练平台（本项目训练端） |
| [17] PyTorch | 深度学习库 | 基础 | 全部生态的底层张量接口 |
| [18] ManiSkill3 | GPU 并行仿真 | 仿真 | 评估端（低显存 3.5GB/128 env） |
| [19] SIMPLER* | 真实到虚拟评估 | 仿真 | 评估协议（MMRV + 排序相关性） |

\* 注意：文件 `[19]_SimpleReNV.pdf` 实际内容是 **SIMPLER** 论文（arXiv 2405.05941），见文末「引用准确性提示」。

---

## 组 B：阶段感知强化学习（项目核心创新）

### HiTS（分层 RL 时间子目标）
MPI-IS Tübingen · NeurIPS 2021 · arXiv:2112.03100
- **贡献**：高层不仅指定"达到什么子目标状态"还指定"何时达到"（时间子目标 `a¹=(g₀,t₀)`），消除 SMDP 转移时间非平稳性，稳定并发生成。
- **方法**：子目标由高层策略**学习生成**（非规则）；事后动作重标定（hindsight relabeling）；低层奖励极稀疏（到达目标状态且恰好 t₀ 步用尽 +1）；"testing transitions" 迫使高层只分配可行子目标。
- **结果**：动态环境 Platforms 成功率 ~86%（HAC 仅 ~40%）。
- **本项目借鉴**：相对时间/倒计时的阶段进度表示、hindsight relabeling 处理阶段分类器与策略联合训练的非平稳性、testing 评估阶段分离器可靠性。

### STARE-VLA（阶段感知 VLA 强化学习）⭐ 最相关工作
TUM / 帝国理工 / 华为慕尼黑 · arXiv:2512.05107v2 · 2025
- **贡献**：StARe（Stage-Aware Reinforcement）即插即用模块——阶段分离器 + 阶段计算器，把长程轨迹按语义阶段拆分并给稠密、可解释、阶段对齐的强化信号；提出 **StA-TPO（离线偏好）+ StA-PPO（在线）** 与 SFT 串成 IPI 三步流水线。
- **方法**：
  - **阶段分离器（监督/规则式）**：由"语义操纵事件"定边界，用末端执行器平移/朝向几何阈值设二元标志。pick-and-place 四阶段 `Reach→Grasp→Transport→Place`；前后阶段"入口条件 = 上一阶段终点条件"保证连续性。
  - **阶段计算器**：① 阶段成本 = 阶段内末端到目标物体的平均欧氏距离；② 势能塑形 `φ = sigmoid(1 − ||e−obj||/L)`，`r'_t = r_t + γφ_{t+1} − φ_t`（不改变最优策略）。
  - StA-TPO：只在上一阶段成功时构造当前阶段偏好对（渐进一致性）；StA-PPO：rollout 在线检测阶段 + 势能塑形后走标准 PPO。
- **结果**：SimplerEnv 平均成功率 **98.0%**（超 RL4VLA 92.5%、SFT 58.3%）；ManiSkill3 **96.4%**（超 RL4VLA 70.5% +25.9 点）；阶段感知对高精度阶段（堆叠 Place、插桩 Upright）去掉后掉点 >20%。
- **本项目关系**：StA-PPO 与本项目"StARe-PPO"结构高度吻合。**关键差异——本项目承诺"无监督阶段拆分"**：STARE-VLA 的分离器是任务专属规则+几何阈值（每任务手工设计），本项目要做的数据驱动无监督拆分（聚类/变点/隐式阶段分配）正是 STARE-VLA 未做、申请书承诺的增量。若 StageDetector 停留在几何阈值，本质就是 STARE-VLA 规则式分离器的变体——"无监督化"是答辩必须讲清的核心竞争力。

### PPO（近端策略优化）
OpenAI · arXiv:1707.06347 · 2017
- **方法**：clip 目标 `L^CLIP = E[min(r(θ)Â, clip(r(θ),1−ε,1+ε)Â)]`；GAE 优势估计 `Â=Σ(γλ)ᵏδ`；总目标 = clip + 价值误差 + 熵。ε=0.2。
- **结果**：MuJoCo 7 任务 1M 步 clip 0.82（最佳）；优于 TRPO/A2C 等。
- **本项目**：在线 RL 算法底座；STARE-VLA 的 StA-PPO = PPO + 阶段塑形，本项目复用 clip/GAE/共享 actor-critic。

### 三方法关系（阶段从哪来、怎么用）

| 维度 | 本项目 StageDetector | STARE-VLA 分离器 | HiTS 子目标 |
|---|---|---|---|
| 阶段来源 | 几何阈值（拟升级无监督） | 任务事件规则+阈值（监督） | 高层策略学习生成 |
| 层级 | 单策略内阶段标注 | VLA 微调模块 | 真正两层 HRL |
| 时间处理 | 可加阶段进度/相位 | 隐式（阶段内势能） | 显式时间子目标 t₀ |
| 用途 | 阶段感知奖励/PPO | 阶段成本+塑形+阶段级偏好 | 高层选"何状态+何时" |

**可借鉴点**：① STARE-VLA——阶段成本当稠密奖励、potential-based shaping、阶段条件成功率指标 `P(stage|stage−1)`、SFT→StA-TPO→StA-PPO 串联；② HiTS——相对时间/倒计时进度、hindsight relabeling、testing 可靠性评估；③ 若写论文，最强贡献证据 = "无监督阶段分离器 + 阶段感知 PPO" 对比 STARE-VLA 规则式分离器（同任务集 SimplerEnv/ManiSkill3，量化免人工阈值后的通用性）。

---

## 组 A：VLA 基础

### RT-2（VLA 开山作）
Google DeepMind · arXiv:2307.15818 · 2023
- 把动作当作"另一种语言"写进 VLM 输出（256 bin 离散成文本 token），Web 知识迁移到机器人控制。
- 结果：seen 91–93%、unseen 62%（RT-1 32%）；55B 需多 TPU 云部署（1–3 Hz）。闭源巨型，仅作范式参照。

### PaLM-E（具身多模态语言模型）
Google / TU Berlin · arXiv:2303.03378 · 2023
- 图像/状态/3D 场景编码注入预训练 LLM（562B）；输出高层规划，底层策略执行。
- TAMP 规划 94.9%（全混合微调）；多任务正迁移、大模型抗遗忘。两级结构 + 巨量，与端到端轻量单策略不符。

### OpenVLA（开源基础模型）⭐ 本项目基础模型
Stanford / UC Berkeley / TRI / Google DeepMind · arXiv:2406.09246 · 2024
- Prismatic-7B（Llama-2 7B + SigLIP/DINOv2 双视觉编码器 + MLP 投影）；动作按分位数定 bin 离散成 256 bin；仅用 Open X-Embodiment 970k 演示做 next-token 预测。
- 结果：29 任务多本体超闭源 RT-2-X **+16.5 点**、参数少 7 倍；**LoRA rank32 只训 1.4% 参数**匹配全量微调、单卡 A100 10–15h、算力省 8×；**int4 量化 71.9%≈bf16 71.3%、显存 7GB**；RTX 4090 ~6Hz。
- **选它理由**：唯一开源可改、可 LoRA/量化轻量微调、消费级显存可跑，与项目诉求完全吻合。短板：未做 Web 联合微调，语义泛化弱于 RT-2 式 co-FT——正是本项目阶段感知强化学习可在目标域补强的点。

### Generalist（RoboVLMs，VLA 设计指南）
清华/字节/NUS/BAAI · arXiv:2412.14058 · 2025
- 600+ 实验系统回答骨干/架构/数据策略：**连续动作优于自回归离散、policy-head 历史融合最优、KosMos 与 PaliGemma 骨干最强、跨本体预训练单独收益有限、域内数据才是关键**（few-shot 预训练 +17.2%）。
- 佐证"OpenVLA 预训练 + 域内阶段感知 RL 微调"路线；为轻量骨干选择提供实证。

---

## 组 C：模型轻量化

### LoRA（低秩微调）
Microsoft Research · arXiv:2106.09685 · 2021
- `W = W0 + BA`（秩 r ≪ d,k），冻结 W0 只训 A/B；部署时合并回 W0，推理零开销。
- GPT-3 175B 可训练参数减少 **10,000×**、显存 1.2TB→350GB、提速 25%；r=1~4 即够。
- **本项目**："消费级显卡可训练"的核心杠杆；与量化/蒸馏正交叠加。

### 轻量化深度模型综述
IJCISIMA Vol.13 · 2021（注意 PDF 文件 xref 损坏，内容系流解压还原）
- 剪枝/蒸馏/量化分类对比；混合模型（KD 蒸馏 → 0.5 剪枝 → FP32→INT8 重训 2 epoch）比原 CNN **小 3 倍、精度仍 97%**。
- 经验：结构化剪枝优先于非结构化、PTQ 开销最小但量化后需 fine-tune 恢复精度。

### 知识蒸馏
Google · arXiv:1503.02531 · 2015
- 软目标 `q_i = exp(z_i/T)/Σ...`；损失 = 高温软目标 CE + 温度 1 真实标签 CE，梯度按 1/T² 缩放需乘 T² 平衡；软目标是强正则。
- MNIST 大网 67 错 → 蒸馏小网 74 错（无软目标 146 错）；3% 数据下软目标 57% vs 硬标签过拟合 44.5%。
- **本项目**：样本效率关键（机器人小数据）；teacher 需离线生成软标签避免重复算力开销。

### TinyVLA（<1B 轻量 VLA）
华东师大/美的 AI Lab · RA-L 2025 · arXiv:2409.12514
- Pythia 小 VLM（70M–1.4B）+ LLaVA 流程；机器人微调仅用 LoRA 更新 Q/K/V（可训练 5%）；**用 Diffusion Policy 头回归连续动作**（不做自回归 token 生成）。
- TinyVLA-H（1.3B，可训练 143M）：真实 Franka 五任务 **94.0% vs OpenVLA 68.3%**、参数少 5.5×；A6000 单步 **14ms vs OpenVLA-7B 292ms（快 20×）**。
- **本项目**：LoRA + 结构重构的直接范例——扩散头替代自回归是比量化更根本的加速；3060/4060 级别可训小 VLA。

### AnoleVLA（SSM 轻量 VLA）
庆应义塾大学 · arXiv:2603.15046 · 2026
- **Mamba（深度状态空间模型）替换 Transformer**，序列长度 O(L) 线性复杂度；单次前向直接出 H=50 步动作 chunk；两阶段训练（速度 L1 → 加速度 L1）。
- ~467M 参数、91 GFLOPs；MetaWorld 67.85%（超 TinyVLA 31.58%）；真实 HSR 5 任务 63%（+21 点）；推理 **216ms/chunk vs 0.5(3B) 578ms**；**单张 RTX 4090 ~20 小时可训完、每任务仅 50 条演示**。
- **本项目**：消费卡可完整训练轻量 VLA 的证明；SSM 线性复杂度是 LoRA 之外的第二维结构优化选项。

### 三级方案（LoRA + INT8/INT4 + 蒸馏）借鉴与风险
- **值得抄**：① TinyVLA 的 LoRA 配方（冻结 backbone、只训 Q/K/V、5% 参数、训练后重参化合并）；② 蒸馏 soft targets **离线缓存**（大机器对演示数据前向一次落盘，学生训练只读缓存）；③ **结构重构优先于量化**（扩散头/Mamba 的加速远大于 INT8）；④ 量化先 PTQ，掉点则量化后 fine-tune 2-3 epoch。
- **风险**：① teacher 算力/显存（7B OpenVLA 在 3060 上单次前向极慢 → 必须离线软标签，或换 1B–3B teacher，且 teacher 与目标场景分布越接近越好）；② INT8 算子 fallback / 校准集 / 激活溢出；3060/4060 非推理卡，TensorRT INT8 收益有限；**INT4 精度风险高，建议先 INT8 保精度**；③ 三级误差累积（蒸馏+量化+LoRA 每级 1-2 点）；④ 小 student 在视角/光照变化下易过拟合（AnoleVLA 主要失败是目标定位）。

---

## 组 D：仿真与基础

### Transformer（Attention Is All You Need）
Google Brain · NIPS 2017
- 缩放点积注意力 + 多头(h=8) + 残差/LayerNorm + 正弦位置编码；WMT14 英德 BLEU 28.4。VLA 主干。

### Dex-Net 4.0（双手抓取）
UC Berkeley AUTOLAB · Science Robotics 2019
- 解析模型合成 500 万抓取样本训练 GQ-CNN（夹爪+吸盘），域随机化 sim-to-real；真机清理料箱 **97%**、>300 MPPH。
- **本项目**：合成数据+域随机化+统一奖励的 sim-to-real 范式；抓取成功率指标可参照。

### Isaac Lab（仿真框架）⭐ 训练端
NVIDIA · arXiv:2511.04831 · 2025
- Isaac Gym 继任者：OpenUSD + PhysX GPU 并行物理 + RTX 渲染 + Tensor API；manager/direct 双工作流；集成 SKRL/RSL-RL/SB3 等。
- 结果：DextrAH >900k FPS、Franka 柜子 >1.6M FPS（16k env/8 卡）；单卡 RTX 5090 接近双卡服务器。
- **本项目**：训练平台承诺有直接支撑——GPU 并行 PPO、teacher-student 蒸馏、视觉域随机化、多相机 tiled 渲染全齐备。

### PyTorch
FAIR · NeurIPS 2019 · arXiv:1912.01703
- 命令式动态图 + 反向自动微分 + CUDA 异步执行；Isaac Lab/ManiSkill3/SIMPLER 全以它为张量接口。

### ManiSkill3（GPU 并行仿真）⭐ 评估端
UCSD 等 · arXiv:2410.00425 · 2025
- 最快开源"状态-视觉"GPU 并行操作仿真器；仿真+渲染全 GPU 并行、异构仿真、12 类任务 20+ 机器人；可评估 Octo/RT-X/RDT-1B 等 VLA。
- 结果：仿真+渲染 >**30,000 FPS**；128 并行环境仅 **3.5GB 显存**（Isaac Lab 14.1GB，低 2-3×）；PickCube RGB 视觉 ~10 分钟收敛；Koch 零样本 sim2real 91.6%；real2sim 评估与真实相关 0.9284。
- **本项目**：8GB 显存限制下比 Isaac Lab 更适合跑视觉 RL 评估；内置 SIMPLER 数字孪生（60-100× 实时）可无监督评估轻量 VLA。

### SIMPLER（真实到虚拟评估）
UCSD/Stanford/UC Berkeley/Google DeepMind · arXiv:2405.05941 · 2024（文件误标为 SimpleReNV）
- 用仿真评估真实数据训练的策略并证明与真实强相关；**MMRV 指标 + 排序相关性（Pearson r）**；绿幕背景 + 纹理/配色匹配 + 系统辨识缩 sim-real 差距。
- 结果：~1500 episode 上 RT-1/RT-2-X/Octo 仿真-真实 r=0.924、MMRV=0.056；单环境渲染 3.5k FPS，比真实评估快 7×。
- **本项目**：评估协议直接用（相对排序而非绝对成功率）；"不造真机也证明真实可迁移性"。

### 组合评估：Isaac Lab 训练 + ManiSkill3/SIMPLER 评估
- **合理且兼容**：三者同以 PhysX + PyTorch 张量为接口，同一套策略推理代码无缝切换；形成"Isaac Lab 训练 → ManiSkill3 评估 → SIMPLER 真实到虚拟协议"闭环。
- **分工**：Isaac Lab 侧重大规模 GPU 并行 RL 训练；ManiSkill3 侧重复用基线 + 异构泛化 + 低显存视觉评估；SIMPLER 提供评估协议。
- **注意**：Isaac Lab（光追渲染，显存/速度重）与 ManiSkill3（SAPIEN 光栅化）非严格同条件，评估需统一分辨率/相机/控制器协议。

---

## 引用准确性提示（答辩/投稿前需核对）

1. **[19] 文件实为 SIMPLER 论文**（*Evaluating Real-World Robot Manipulation Policies in Simulation*, arXiv:2405.05941）。申请书参考文献 [19] 写作 "SimpleReNV: A Simple Real-to-Virtual Benchmark for Robotic Manipulation"，作者 "Xu S, Naderi A, Terenin A"——**标题与作者与 PDF 内容不符**（SIMPLER 实际作者为 Xiaomeng Xu、Huy Ha、Calvin Luo 等，论文内无 Terenin/Naderi 作为该文作者）。建议核对 arXiv 2405.05941 真实标题与作者，修正申请书引用。
2. **[11] 文件名 "Towards Generalist Robot Policies" 与内容不符**：实际论文为 *What Matters in Building Vision-Language-Action Models for Generalist Robots*（arXiv:2412.14058）。申请书 [11] 标题与之近似但措辞略异，建议统一。
3. **[7] 轻量化综述 PDF xref 损坏**：已通过流解压还原内容；引用时以论文正式题录为准（IJCISIMA Vol.13, 2021, pp.110-129，作者 Musa 等）。
4. **目录缺 [15]（Gazebo）**：申请书技术路线提到 Gazebo，但论文库无对应 PDF；且 Gazebo 从未被旧工程代码引用（仿真只用了 Isaac Sim）。答辩注意一致性。

---

## 对项目落地的关键结论（浓缩）

1. **基础模型用 OpenVLA**，配合 LoRA（只训 5% 参数）+ INT4 即可在消费级显卡微调；后续可换 PaliGemma/KosMos 等更小骨干或 Mamba 结构（AnoleVLA）。
2. **核心竞争力是"无监督阶段拆分"**：STARE-VLA 已证明"阶段感知 RL + VLA"可行（98%/96.4%），但它用手工规则分离器。项目把 StageDetector 从几何阈值推进到数据驱动无监督分离，就是区别于最相关工作、答辩能站住脚的增量。
3. **轻量化落地顺序**：先保证非自回归动作头（扩散/Mamba），再做蒸馏（离线软标签）+ LoRA；INT8 保精度，INT4 作可选激进档。
4. **仿真闭环**：Isaac Lab 训练（GPU 并行 PPO + 阶段奖励）→ ManiSkill3 评估（低显存、可跑 VLA）→ SIMPLER 协议证明真实可迁移性。
