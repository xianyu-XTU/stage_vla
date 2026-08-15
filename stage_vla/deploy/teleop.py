"""teleop.py —— 人工遥操（M3 演示数据采集用）。

M3 演示链：teleop/rollout → ``data.demo_record.record_episode`` → ``outputs/demos/*.npz``
→ ``finetune_lora.py``。旧工程完全没有演示数据生产脚本，本模块补上该环节。
"""

from __future__ import annotations


def teleop_episode(env, n_steps: int = 200) -> list:
    """人工遥操采集一段轨迹（M3 实现，返回逐帧动作）。"""
    raise NotImplementedError("人工遥操为 M3 里程碑实现。")
