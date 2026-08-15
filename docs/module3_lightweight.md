# 模块③ 模型轻量化

## 目标（申请书承诺）

LoRA 低秩微调 + INT8/INT4 量化 + 知识蒸馏组合技术，全维度轻量化，实现消费级显卡
（RTX 3060/4060）训练与端侧部署。

## 子模块

| 子模块 | 文件 | 状态 |
|---|---|---|
| QLoRA 低秩微调（冻结主干，只训低秩矩阵） | `lightweight/lora.py` | ⏳ M3 |
| 运行时量化（INT4/INT8） | `lightweight/quantize.py` | ⏳ M3；**INT8 不造假** |
| 知识蒸馏（teacher-student） | `lightweight/distill.py` | ⏳ M3 |
| 轻量化指标（参数量/显存/延迟） | `lightweight/metrics.py` | ✅ 已实现 |

## 演示数据链（补齐旧工程断链）

旧工程 `outputs/demos/*.npz` 没有任何脚本生产，导致 QLoRA 微调无法运行。M3 补全：

```
collect_demos.py → outputs/demos/episode_*.npz（image/instruction/action）
                → finetune_lora.py（只存 adapter）
```

## INT8 纪律

**绝不复用旧工程"INT8 静默走 bf16"的假量化**。`quantize(bits=8)` 在接入
optimum GPTQ/AWQ 或 torch.ao 之前一律抛 `QuantizationUnsupported`。
