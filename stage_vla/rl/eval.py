"""eval.py —— 阶段感知 PPO 评估（**需 Isaac 环境**）。

加载 checkpoint，rollout 用阶段检测器判 grasp / stack 成功率。
v1（M0~M1）用几何阶段检测；M2 融合后支持"VLA 闭环 + 阶段反馈"评估。
"""

from __future__ import annotations

from pathlib import Path


def find_checkpoint(load_run: str, log_root: str = "logs") -> Path:
    """按运行目录前缀找最新 checkpoint（``model_*.pt``）。

    Args:
        load_run: 运行目录名（前缀匹配）
        log_root: 日志根目录

    Raises:
        FileNotFoundError: 未找到匹配的 checkpoint
    """
    run_dir = None
    root = Path(log_root)
    if root.is_dir():
        candidates = sorted(root.glob(f"{load_run}*"))
        if candidates:
            run_dir = candidates[-1]
    if run_dir is None:
        raise FileNotFoundError(f"未找到运行目录：{log_root}/{load_run}*")

    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"运行目录中无 checkpoint：{run_dir}")
    return checkpoints[-1]


def evaluate(
    settings,
    load_run: str,
    num_episodes: int = 10,
    max_steps: int = 500,
    headless: bool = True,
) -> dict:
    """评估已训练策略，返回成功率统计（**需 Isaac 环境**）。

    Returns:
        {"success_rate": float, "grasp_rate": float, "avg_episode_len": float, ...}
    """
    raise NotImplementedError("evaluate 为 M1 里程碑实现（v1 状态版 rollout + 阶段检测判成功）。")
