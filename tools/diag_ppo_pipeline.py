"""诊断：直接复用 ppo_loop 的 _collect_rollout/_compute_gae/_ppo_update，定位崩溃点。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_vision_env_cfg  # noqa: E402
from stage_vla.rl.action_interface import IKRelActionInterface  # noqa: E402
from stage_vla.rl.online_feedback import StageFeedback  # noqa: E402
from stage_vla.rl.ppo_loop import _collect_rollout, _compute_gae, _ppo_update  # noqa: E402
from stage_vla.rl.runner import create_raw_env  # noqa: E402
from stage_vla.rl.vla_policy import VisionFeatureExtractor, VisionOnlyPPOPolicy  # noqa: E402

settings = load_settings()
stages = list(settings.stages)
env_cfg = build_vision_env_cfg(settings, num_envs=8, seed=42, cam_res=96)
env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)

fe = VisionFeatureExtractor(settings, device="cuda")
policy = VisionOnlyPPOPolicy(features_dim=fe.features_dim, feature_extractor=fe,
                             stage_feedback_dim=len(stages))
iface = IKRelActionInterface()
feedback = StageFeedback(stages)

print("=== _collect_rollout ===")
rollout = _collect_rollout(env, policy, iface, feedback, stages, horizon=24)
print(f"[OK] rollout keys={list(rollout.keys())} x={tuple(rollout['x'].shape)} x.device={rollout['x'].device}")
try:
    torch.cuda.synchronize()
    print("[sync] 收集后同步 OK（无 pending 错误）")
except Exception as e:
    print(f"[FAIL] 收集后同步即报错: {e}")
    env.close(); sim_app.close(); sys.exit(1)

print("=== _compute_gae ===")
adv, ret = _compute_gae(rollout, gamma=0.99, lam=0.95)
print(f"[OK] adv={tuple(adv.shape)} isfinite={torch.isfinite(adv).all().item()}")

print("=== _ppo_update ===")
optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=1e-3)
losses = _ppo_update(policy, optimizer, rollout, adv, ret, ppo_epochs=5, mini_batches=4, clip=0.2)
print(f"[OK] update losses={losses}")
env.close()
sim_app.close()
print("=== 全管线稳定 ===")
