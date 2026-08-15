# 模块① 长程任务阶段感知机制

## 目标

把"抓取-移动-放置-堆叠"长程任务拆成语义明确、顺序固定的子阶段，为每步操作提供
稠密奖励信号，解决奖励稀疏与信用分配难题。

## 子模块

| 子模块 | 文件 | 状态 |
|---|---|---|
| 语义分离（指令 → 子目标 → 阶段序列） | `stage_vla/stages/semantic.py` | ✅ 规则版（预留 LLM 解析接口） |
| 阶段计算器（信号 → 阶段索引/进度，单一实现） | `stage_vla/stages/calculator.py` | ✅ 纯函数 |
| 几何阶段检测器（原语 + 委托 calculator） | `stage_vla/stages/detector.py` | ✅ 向量化 |
| 无监督阶段分离器（脱离人工标注） | `stage_vla/stages/unsupervised.py` | ✅ M1（过分割+进度排序+贪心合并，合成轨迹一致性 >0.7） |
| 稠密奖励（势能塑形 + 阶段完成奖） | `stage_vla/stages/rewards.py` + `rewards_isaac.py` | ✅ 纯函数 + Isaac 适配 |

## 奖励设计要点（回归点）

1. **势能塑形**：`r = γ·φ(s_t) − φ(s_{t−1})`，φ = 阶段进度和。进步给正，退步给负。
2. **阶段完成奖**：跨入新阶段一次性给权重（approach=0 / grasp=2 / lift=1 / move=0.5 / stack=10）。
3. **首帧处理**：用 `episode_length_buf == 1` 识别 reset 首帧，置零该帧塑形/完成奖。
4. **dt 校正**：Isaac 侧返回 `值 / env.step_dt` 抵消 manager 的 `*dt`，config 权重即每步名义值。
5. **阶段判定逆序链**：stack → move → lift → grasp → approach；`is_stacked` 独立于 `is_grasped`。
