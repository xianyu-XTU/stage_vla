"""plot_curves.py —— 从训练日志解析指标并绘制训练曲线。

解析 rsl_rl 训练日志（outputs/train_curve.log）中每迭代的：
- Mean reward / Episode_Reward/stage_progress / Episode_Reward/stage_transition
- Episode_Termination/success

绘制成曲线图保存到 outputs/training_curves.png。

用法::

    <isaac_sim>\\kit\\python\\python.exe tools\\plot_curves.py [log_path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "outputs" / "train_curve.log"

METRICS = {
    "mean_reward": re.compile(r"Mean reward:\s+([-\d.]+)"),
    "stage_progress": re.compile(r"Episode_Reward/stage_progress:\s+([-\d.]+)"),
    "stage_transition": re.compile(r"Episode_Reward/stage_transition:\s+([-\d.]+)"),
    "lifting_object": re.compile(r"Episode_Reward/lifting_object:\s+([-\d.]+)"),
    "success": re.compile(r"Episode_Termination/success:\s+([\d.]+)"),
}


def parse(log: Path) -> dict[str, list[float]]:
    """按迭代对齐解析：每种指标出现的第 N 次对应第 N 迭代。"""
    text = log.read_text(encoding="utf-8", errors="replace")
    data = {k: [] for k in METRICS}
    for k, pattern in METRICS.items():
        data[k] = [float(m) for m in pattern.findall(text)]
    return data


def main() -> int:
    if not log_path.is_file():
        print(f"[plot] 找不到日志：{log_path}")
        return 1
    data = parse(log_path)
    n = len(data["mean_reward"])
    print(f"[plot] 解析到 {n} 个迭代指标", flush=True)
    for k, v in data.items():
        print(f"  {k}: {len(v)} 个点, 末值 {v[-1] if v else 'N/A'}", flush=True)
    if n < 3:
        print("[plot] 迭代点太少，跳过绘图")
        return 0

    import matplotlib
    matplotlib.use("Agg")   # 无显示环境
    import matplotlib.pyplot as plt

    iters = range(n)
    n_plots = len(plots)
    ncols = 2
    nrows = (n_plots + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    fig.suptitle("StARe-PPO Training Curves (Stage-Aware RL + Lift Reward)", fontsize=14)
    axes = [ax for row in axes for ax in row]  # 展平

    def smooth(xs, window=10):
        if len(xs) < window:
            return xs
        import numpy as np
        a = np.convolve(xs, np.ones(window) / window, mode="valid")
        return a.tolist()

    # 动态选择有数据的指标（优先显示核心项）
    all_plots = [
        ("mean_reward", "Mean reward", "cumulative reward"),
        ("stage_transition", "stage_transition", "stage transitions"),
        ("lifting_object", "lifting_object", "cube lifted reward"),
        ("stage_progress", "stage_progress", "potential shaping"),
        ("success", "success rate", "success"),
    ]
    plots = [(k, t, y) for k, t, y in all_plots if data.get(k)]
    if not plots:
        print("[plot] 无可用指标")
        return 0
    for ax, (key, title, ylabel) in zip(axes.flat, plots):
        vals = data[key]
        if not vals:
            ax.set_title(f"{title}: no data")
            continue
        ax.plot(iters, vals, alpha=0.3, color="tab:blue", label="raw")
        s = smooth(vals)
        off = len(vals) - len(s)
        ax.plot(range(off, len(vals)), s, color="tab:blue", linewidth=2, label="smoothed(10)")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = _ROOT / "outputs" / "training_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"[plot] 已保存曲线 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
