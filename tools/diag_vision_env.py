"""诊断：视觉环境用零动作能否稳定 step（隔离是渲染/物理问题还是策略动作问题）。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402

settings = load_settings()
env_cfg = build_vision_env_cfg(settings, num_envs=4, seed=42, cam_res=96)
env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)

obs, _ = env.reset()
print("=== obs keys ===", list(obs["policy"].keys()))
img = obs["policy"]["table_cam"]
print("table_cam:", type(img).__name__, getattr(img, "shape", None), getattr(img, "dtype", None))

inner = env.unwrapped
zero = torch.zeros(env.action_space.shape, device=inner.device)
print(f"=== 零动作 step 5 次 ===  action_space={env.action_space.shape}")
for i in range(5):
    obs, rew, term, trunc, _ = env.step(zero)
    img = obs["policy"]["table_cam"]
    print(f"step {i}: rew={rew.tolist()} img[0]={img[0].shape} {img.dtype}")
    if bool((term | trunc).any()):
        print("  (有终止)")
print("[OK] 零动作 5 步稳定")
env.close()
sim_app.close()
