"""诊断：真实相机图像 → 视觉塔特征 → 动作，是否产生 NaN / 是否物理崩溃。"""
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
from stage_vla.rl.runner import create_raw_env  # noqa: E402
from stage_vla.rl.vla_policy import VisionFeatureExtractor, VisionOnlyPPOPolicy  # noqa: E402

N_ENVS = 8
N_STEPS = 24
settings = load_settings()
env_cfg = build_vision_env_cfg(settings, num_envs=N_ENVS, seed=42, cam_res=96)
env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)

fe = VisionFeatureExtractor(settings)
policy = VisionOnlyPPOPolicy(features_dim=fe.features_dim, feature_extractor=fe, stage_feedback_dim=len(settings.stages), device="cuda")
iface = IKRelActionInterface()
feedback = StageFeedback(list(settings.stages))

obs, _ = env.reset()
img = obs["policy"]["table_cam"]
print(f"图像: {img.shape} {img.dtype}  min={img.min().item()} max={img.max().item()}")

fb = feedback.feedback()
if fb is None:
    fb = torch.zeros(img.shape[0], len(settings.stages), device="cuda")
x = policy.features(img, fb)
print(f"特征: {tuple(x.shape)}  isfinite={torch.isfinite(x).all().item()}  "
      f"mean={x.mean().item():.3f} std={x.std().item():.3f}")

action, log_prob, value = policy.sample_from_x(x)
print(f"动作: isfinite={torch.isfinite(action).all().item()} mean={action.mean().item():.3f}")
env_action = iface.to_env_action(action)
print(f"env动作: isfinite={torch.isfinite(env_action).all().item()} min={env_action.min().item():.3f} max={env_action.max().item():.3f}")

print(f"=== 用策略动作 step {N_STEPS} 次（训练同配置）===")
for i in range(N_STEPS):
    torch.cuda.synchronize()
    obs, rew, term, trunc, _ = env.step(env_action)
    if i % 4 == 0:
        print(f"step {i}: rew={rew.tolist()}")
    img = obs["policy"]["table_cam"]
    fb = feedback.feedback() or torch.zeros(img.shape[0], len(settings.stages), device="cuda")
    x = policy.features(img, fb)
    action, log_prob, value = policy.sample_from_x(x)
    env_action = iface.to_env_action(action)
    torch.cuda.synchronize()
print(f"[OK] 策略动作 step {N_STEPS} 次稳定")
env.close()
sim_app.close()
