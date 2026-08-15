"""验证 VisionFeatureExtractor 加载 + 前向（诊断用，无需 Isaac Sim）。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.rl.vla_policy import VisionFeatureExtractor, VisionOnlyPPOPolicy  # noqa: E402


def main() -> int:
    settings = load_settings()
    print(f"vision_only_dir = {settings.require_path('vision_only_dir')}")

    fe = VisionFeatureExtractor(settings)
    print(f"features_dim = {fe.features_dim}")
    print(f"加载后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 合成图像 [200,200,3] uint8
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    feats = fe(img)
    print(f"特征形状: {tuple(feats.shape)} 显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    assert feats.shape[-1] == fe.features_dim

    # 顶部策略：act 传入原始图像（策略内部特征提取器处理）
    policy = VisionOnlyPPOPolicy(features_dim=fe.features_dim, feature_extractor=fe, stage_feedback_dim=5, device="cuda")
    feedback = torch.rand(1, 5)
    action, log_prob, value = policy.act(img, stage_feedback=feedback)
    print(f"action={tuple(action.shape)} log_prob={log_prob.item():.3f} value={value.item():.3f} "
          f"总显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    n = sum(p.numel() for p in policy.trainable_parameters())
    print(f"可训练参数: {n/1e6:.2f}M")
    assert action.shape == (1, 7)
    print("[OK] VisionFeatureExtractor + PPO 策略 加载/前向验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
