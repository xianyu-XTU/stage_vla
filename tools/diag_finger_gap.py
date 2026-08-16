"""diag_finger_gap.py —— 实测 Franka 手指在世界系的实际张距（对照方块尺寸）。

在多个夹爪关节位置下读取两个手指 link 的世界坐标，计算世界间距，
看 0.019 关节值时开口多大、能否包住 5cm 方块。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0


def main() -> int:
    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=1, seed=42, cam_res=96)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)
    inner = env.unwrapped
    dev = inner.device
    env.reset()

    # 找手指 link 名
    body_names = inner.scene["robot"].data.body_names
    finger_names = [n for n in body_names if "finger" in n or "finger" in n.lower()]
    print(f"[gap] 手指相关 link: {finger_names}", flush=True)
    lf = next((n for n in body_names if "leftfinger" in n.lower()), finger_names[0] if finger_names else None)
    rf = next((n for n in body_names if "rightfinger" in n.lower()), finger_names[1] if len(finger_names) > 1 else None)
    lf_id, rf_id = body_names.index(lf), body_names.index(rf)

    def finger_gap() -> float:
        pos = inner.scene["robot"].data.body_pos_w[:, [lf_id, rf_id], :]  # [1,2,3]
        return torch.norm(pos[0, 0] - pos[0, 1]).item()

    def finger_joint() -> float:
        return inner.scene["robot"].data.joint_pos[0, 7].item()

    # 测量几个关节位置下的世界张距
    print(f"[gap] 方块尺寸参考: 0.05m (5cm)", flush=True)
    # 初始（开）
    print(f"[gap] 初始: joint={finger_joint():.4f} 手指link间距={finger_gap()*1000:.1f}mm", flush=True)
    # 闭合到 0.019（之前抓取停住的位置）
    act = torch.zeros(1, 7, device=dev)
    act[:, 6] = GRIPPER_CLOSE
    for _ in range(5):
        env.step(act)
    print(f"[gap] 闭合5步: joint={finger_joint():.4f} 手指link间距={finger_gap()*1000:.1f}mm", flush=True)
    for _ in range(40):
        env.step(act)
    print(f"[gap] 闭合45步: joint={finger_joint():.4f} 手指link间距={finger_gap()*1000:.1f}mm", flush=True)

    # 完全打开
    act[:, 6] = GRIPPER_OPEN
    for _ in range(20):
        env.step(act)
    print(f"[gap] 全开: joint={finger_joint():.4f} 手指link间距={finger_gap()*1000:.1f}mm", flush=True)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
