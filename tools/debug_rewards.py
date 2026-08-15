"""debug_rewards.py —— 阶段感知奖励诊断（**需 Isaac 环境**）。

打印每步的原始几何信号（末端/方块位置）、阶段进度与两类阶段感知奖励值，
用于验证奖励机制是否真正生效（曾有坐标帧 bug 导致恒 0）。::

    python tools\\run_isaaclab.py tools\\debug_rewards.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.envs.cfg_surgery import build_stage_env_cfg  # noqa: E402
from stage_vla.rl.runner import create_env  # noqa: E402

settings = load_settings()
env_cfg = build_stage_env_cfg(settings, num_envs=4, seed=42)
env, sim_app = create_env(settings.task["id_state"], env_cfg, headless=True)

from stage_vla.stages.rewards_isaac import (  # noqa: E402
    compute_stage_progress,
    stage_potential_reward,
    stage_transition_reward,
)

from stage_vla.stages.rewards_isaac import _stage_signals_from_env  # noqa: E402

obs, _ = env.reset()
# reward func 收到的 env 是内层 ManagerBasedRLEnv（有 .scene）
inner = env.unwrapped
print("=== 原始信号（env[0]）===")
print(f"env_origins[0] = {inner.scene.env_origins[0].tolist()}")
# 两种读法对比：root_pos_w 是否已经是世界坐标
for name in ("cube_1", "cube_2", "cube_3"):
    rpw = inner.scene[name].data.root_pos_w[0]
    print(f"{name} root_pos_w={rpw.tolist()}  +origins={ (rpw + inner.scene.env_origins[0]).tolist() }")
print(f"ee_frame target_pos_w[0]={inner.scene['ee_frame'].data.target_pos_w[0].tolist()}")
print(f"ee_frame target_pos_w[0][0]={inner.scene['ee_frame'].data.target_pos_w[0][0].tolist()}")
ee_w, cube_pos, grasp, stacked = _stage_signals_from_env(inner)
print(f"ee_w[0]     = {ee_w[0].tolist()}")
for name, pos in cube_pos.items():
    print(f"cube_{name}   = {pos[0].tolist()}")

# obs 交叉验证（env-local 观测）
obs0 = obs["policy"][0]
print(f"obs cube_positions (64:73) = {obs0[64:73].tolist()}")
print(f"obs eef_pos (85:88)        = {obs0[85:88].tolist()}")
print(f"obs joint_pos (7:16)       = {obs0[7:16].tolist()}")
print(f"grasp_sig   = {grasp.tolist()}")
print(f"stacked_sig = {stacked.tolist()}")

print("=== 逐步 step（4 环境）===")
for i in range(6):
    action = torch.rand(env.action_space.shape) * 0.05
    obs, rew, dones, infos = env.step(action)
    prog = compute_stage_progress(inner)          # [N, n_stages]
    sp = stage_potential_reward(inner, weights=settings.reward_weights)
    st = stage_transition_reward(inner, weights=settings.reward_weights)
    print(f"step {i}: rew={rew.tolist()}")
    print(f"        prog_sum={prog.sum(dim=1).tolist()}  prog[0]={prog[0].tolist()}")
    print(f"        stage_progress={sp.tolist()}  stage_transition={st.tolist()}")

sim_app.close()
