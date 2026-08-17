"""state_bank.py —— 阶段成功状态库（交接文档 11 节：stage-success state bank）。

解决"阶段 reset 分布不连续"：lift/move/stack 训练仍从桌面默认态学，但部署时收到
上一阶段真实成功后的状态（夹持姿态/速度/接触），``p_train(s0) != p_deploy(s)``，
导致 grasp 单独能稳定、一切换 lift 就掉块。

机制（默认不启用，需 Isaac 环境验证后开长训）：
- **采集**：``rewards_isaac._grasp_state``（同帧幂等）在稳定抓取**首次达标**时置
  ``env.extras["_stage_grasp_just_stable"]``；``capture_stage_states`` 事件
  （mode="env_step"）读到该标志即把完整状态存入 ``StageStateBank``；
- **重置**：``reset_from_stage_bank`` 事件（mode="reset"）以 ``ratio`` 概率从库采样
  并 ``write_*_to_sim_index`` 写回 sim（root_pos_w 已是世界系，可直接写），
  其余用默认初始化 —— 80/20 混合课程。

用法::

    # 1) 训练 grasp 并采集状态库
    python tools\\run_isaaclab.py scripts\\train_stage.py --stage grasp --state_bank
    # 2) 训练 lift，80% 从 grasp 成功态 reset
    python tools\\run_isaaclab.py scripts\\train_stage.py --stage lift \\
        --bank_from logs/stage_grasp/stage_grasp_bank.pt --bank_ratio 0.8

本模块顶层只 import torch/random（纯部分可无 Isaac 单测）；Isaac 写回在函数内。
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

# 场景中需要随成功态一并写回的方块（抓取目标 / 底座 / 干扰）
_CUBE_NAMES = ("cube_2", "cube_1", "cube_3")


class StageStateBank:
    """按阶段存储完整状态样本（robot 关节 + 各方块 root pose/vel，CPU）。"""

    def __init__(self, max_capacity: int = 1000):
        self._states: dict[str, list[dict]] = {}
        self.max_capacity = max_capacity

    # ------------------------------------------------------------------
    def add(self, stage: str, state: dict) -> None:
        lst = self._states.setdefault(stage, [])
        lst.append(state)
        if len(lst) > self.max_capacity:
            del lst[: len(lst) - self.max_capacity]

    def count(self, stage: str | None = None) -> int:
        if stage is None:
            return sum(len(v) for v in self._states.values())
        return len(self._states.get(stage, []))

    def sample(self, stage: str, device) -> dict | None:
        lst = self._states.get(stage)
        if not lst:
            return None
        state = random.choice(lst)
        return {k: v.to(device) for k, v in state.items()}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"max_capacity": self.max_capacity, "states": self._states}, str(path))

    def load(self, path: Path) -> None:
        data = torch.load(str(path), map_location="cpu")
        self._states = data["states"]
        self.max_capacity = data.get("max_capacity", self.max_capacity)

    def __repr__(self) -> str:
        return f"<StageStateBank {self.count()} states>"


# ============================================================================
# 状态采集 / 写回（**需 Isaac 环境**；函数内惰性访问 env.scene）
# ============================================================================
def _capture_states(env, env_ids: torch.Tensor, bank: StageStateBank, stage: str) -> None:
    """把 ``env_ids`` 指定的环境状态存入 bank（读 robot + 三方块，CPU 拷贝）。"""
    robot = env.scene["robot"]
    batch: dict[str, torch.Tensor] = {
        "joint_pos": robot.data.joint_pos.torch[env_ids].cpu().clone(),
        "joint_vel": robot.data.joint_vel.torch[env_ids].cpu().clone(),
    }
    for name in _CUBE_NAMES:
        obj = env.scene[name]
        batch[f"{name}_pos"] = obj.data.root_pos_w.torch[env_ids].cpu().clone()
        batch[f"{name}_quat"] = obj.data.root_quat_w.torch[env_ids].cpu().clone()
        batch[f"{name}_lin_vel"] = obj.data.root_lin_vel_w.torch[env_ids].cpu().clone()
        batch[f"{name}_ang_vel"] = obj.data.root_ang_vel_w.torch[env_ids].cpu().clone()
    for i, e in enumerate(env_ids.tolist()):
        bank.add(stage, {k: v[i].clone() for k, v in batch.items()})


def capture_stage_states(env, env_ids, bank: StageStateBank, stage: str,
                         flag_key: str = "_stage_grasp_just_stable") -> None:
    """EventTerm（mode="env_step"）：读取成功标志，把达标环境的状态存入 bank。

    ``rewards_isaac._grasp_state`` 在稳定抓取首次达标时置 ``flag_key``；本函数每步
    廉价检查，命中才采集（一次 episode 通常只触发一次）。
    """
    env = getattr(env, "unwrapped", env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    flag = env.extras.get(flag_key)
    if flag is None:
        return
    ready = flag.to(env.device)
    cap_ids = env_ids[ready[env_ids]]
    if len(cap_ids) == 0:
        return
    _capture_states(env, cap_ids, bank, stage)


def reset_from_stage_bank(env, env_ids, bank: StageStateBank, stage: str, ratio: float = 0.8) -> None:
    """EventTerm（mode="reset"）：以 ``ratio`` 概率从 bank 采样初始化 reset 环境。

    采样状态为上一阶段真实成功态（与部署时 grasp 结束后的状态分布一致），
    其余环境走默认初始化（保底、防过拟合单一初始态）。
    """
    env = getattr(env, "unwrapped", env)
    lst = bank._states.get(stage)
    if not lst or env_ids is None:
        return
    for e in env_ids.tolist():
        if random.random() > ratio:
            continue
        _apply_state(env, e, random.choice(lst))


def _apply_state(env, env_id: int, state: dict) -> None:
    """把一个状态写回 sim（root_pos_w 已是世界系，直接写，勿加 env_origins）。"""
    dev = env.device
    eidx = torch.tensor([env_id], device=dev)
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim_index(
        joint_pos=state["joint_pos"].unsqueeze(0).to(dev),
        joint_vel=state["joint_vel"].unsqueeze(0).to(dev),
        env_ids=eidx,
    )
    for name in _CUBE_NAMES:
        obj = env.scene[name]
        pose = torch.cat([state[f"{name}_pos"], state[f"{name}_quat"]]).unsqueeze(0).to(dev)
        vel = torch.cat([state[f"{name}_lin_vel"], state[f"{name}_ang_vel"]]).unsqueeze(0).to(dev)
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=eidx)
        obj.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=eidx)
