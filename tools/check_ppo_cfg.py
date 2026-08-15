"""check_ppo_cfg.py —— 验证 PPO 配置能否在已安装 rsl_rl 上构造 + 迁移（**需 Isaac 环境**）。

训练路径（cfg→runner→OnPolicyRunner）最容易因 rsl_rl 版本 API 变化悄悄坏掉。
本工具不启动仿真，快速验证配置可构造、废弃字段迁移可过。::

    python tools\\run_isaaclab.py tools\\check_ppo_cfg.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.config import load_settings
from stage_vla.rl.cfg import build_runner_cfg

settings = load_settings()
print("[1] build_runner_cfg ...")
cfg = build_runner_cfg(settings.ppo)
print(f"[OK] 构造成功 experiment_name={cfg.experiment_name}")

print("[2] handle_deprecated_rsl_rl_cfg ...")
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
import importlib.metadata
cfg = handle_deprecated_rsl_rl_cfg(cfg, importlib.metadata.version("rsl-rl-lib"))
print(f"[OK] 迁移成功 algorithm={cfg.algorithm.class_name}")
