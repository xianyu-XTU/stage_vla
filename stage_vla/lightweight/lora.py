"""lora.py —— LoRA/QLoRA 低秩微调（模块③，M3 落地）。

申请书承诺：采用 LoRA 低秩微调技术，冻结模型主干，仅训练低秩矩阵，大幅减少训练参数。

流程（M3 落地）：
    load_base_4bit → prepare_qlora（peft）→ make_demo_loader（消费 outputs/demos/*.npz）
    → train_qlora（只对动作 token 算 loss，只存 adapter）

M0 只定义入口函数，抛指引性错误。
"""

from __future__ import annotations


def finetune(settings, demo_dir: str, epochs: int = 1, out_dir: str = "outputs/checkpoints/lora") -> None:
    """QLoRA 微调入口（M3 实现）。

    Raises:
        NotImplementedError: M3 里程碑实现。
    """
    raise NotImplementedError(
        "QLoRA 微调为 M3 里程碑。演示链：collect_demos.py → outputs/demos/*.npz → 本函数。"
    )
