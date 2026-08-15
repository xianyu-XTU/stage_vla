"""demo_record.py —— 演示数据采集（补齐旧工程断链，M3 落地）。

旧工程 QLoRA 微调期待 ``outputs/demos/*.npz``（含 ``{image, instruction, action}``），
但**没有任何脚本生产这种文件**，导致微调管线无法运行。本模块定义采集接口：

    record_episode(env, policy, instruction, out_dir, n_steps)
      → 写入 outputs/demos/episode_<ts>.npz（image / instruction / action）

采集来源：人工遥操（teleop）、RL 最优策略 rollback、或演示脚本回放。
M0 只定义接口；M3 落地并用脚本 ``scripts/collect_demos.py`` 接入。
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import DemoNotFoundError


class DemoEpisode:
    """一段演示：对齐的图像 / 指令 / 动作。

    npz 文件格式约定（M3 落地后严格一致）：:
        image:       [T, H, W, 3]  uint8
        instruction: str
        action:      [T, action_dim]  float32
        meta:        dict（策略名 / 时间戳 / 是否人工遥操）
    """

    def __init__(self, image, instruction: str, action, meta: dict | None = None):
        self.image = image
        self.instruction = instruction
        self.action = action
        self.meta = meta or {}


def record_episode(
    env,
    policy,
    instruction: str,
    out_dir: str | Path = "outputs/demos",
    n_steps: int = 200,
) -> Path:
    """采集一段演示并写入 ``out_dir/episode_*.npz``（M3 实现）。

    Raises:
        NotImplementedError: M3 里程碑实现。
    """
    raise NotImplementedError(
        "演示数据采集为 M3 里程碑。采集链：collect_demos.py → outputs/demos/*.npz → finetune_lora.py。"
    )


def load_demo_file(path: str | Path) -> DemoEpisode:
    """从 npz 加载一段演示（M3 实现）。

    Raises:
        DemoNotFoundError: 文件缺失或格式不符。
    """
    path = Path(path)
    if not path.is_file():
        raise DemoNotFoundError(f"演示文件不存在：{path}")
    raise NotImplementedError("演示数据加载为 M3 里程碑实现。")
