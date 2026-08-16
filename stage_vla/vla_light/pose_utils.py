"""pose_utils.py —— 末端位姿控制辅助（四元数运算 + 姿态对齐）。

为脚本化抓取提供"手指朝下"姿态控制：读取末端当前四元数 → 计算把末端 z 轴旋转到
世界 -z（朝下）所需的轴角增量 → 经 IK 相对动作喂给仿真。
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭（wxyz）。"""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """四元数相乘（wxyz）。"""
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return torch.cat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用四元数旋转向量 v（v 为 [3]，自动广播到 q 的 batch）。"""
    vv = v.to(q.device).expand(*q.shape[:-1], 3)
    qv = torch.cat([torch.zeros_like(q[..., :1]), vv], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def axis_angle_to_delta(cur_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """当前到目标的旋转增量，表示为**末端本地坐标系的轴角**。

    - 世界系旋转增量：``delta_world = target * cur⁻¹``
    - 转到末端本地系：``delta_local = cur⁻¹ * delta_world * cur`` 的向量部分 × 2*角度
    """
    d_world = quat_mul(target_quat, quat_conj(cur_quat))
    # 转到末端本地系
    d_local = quat_mul(quat_mul(quat_conj(cur_quat), d_world), cur_quat)
    # 轴角 = 2 * atan2(|v|, w) 沿 v 轴
    v = d_local[..., 1:]
    w = d_local[..., 0:1]
    angle = 2.0 * torch.atan2(v.norm(dim=-1, keepdim=True).clamp(min=_EPS), w.abs())
    axis = v / v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return axis * angle


def orient_ee_to_down(env, max_steps: int = 60, tol_deg: float = 8.0, tag: str = ""):
    """把末端旋转到"手指朝下"（末端 z 轴 → 世界 -z）。

    **姿态增量约定（查源码确认）**：``apply_delta_pose`` 把动作的 3 维姿态增量解释为
    **父坐标系轴角**，经 ``target = quat_mul(quat_from_angle_axis(axis, angle), source)``
    **左乘**到当前 ee 姿态。因此正确的增量 = 世界系轴角 ``cross(ee_z, -z) * angle``
    （不需要转换到末端本地系——旧实现错了）。四元数用 Isaac Lab 数学函数（xyzw），
    场景数据是 wxyz，需转换。
    """
    from isaaclab.utils.math import quat_apply  # xyzw 约定

    inner = env.unwrapped
    dev = inner.device
    z_down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    z_up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    for step in range(max_steps):
        q_wxyz = inner.scene["ee_frame"].data.target_quat_w[:, 0, :]        # [1,4] wxyz
        q_xyzw = q_wxyz[..., [3, 0, 1, 2]]                                  # → xyzw
        z_world = quat_apply(q_xyzw, z_up)                                  # 末端 z 在世界系
        z_world = z_world / z_world.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        cos = (z_world * z_down).sum(-1).clamp(-1, 1)
        ang = torch.acos(cos)
        if ang.item() * 57.3 < tol_deg:
            break
        axis = torch.cross(z_world, z_down.unsqueeze(0))                    # 世界系旋转轴
        axis_n = axis / axis.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        rot_vec = axis_n * ang.unsqueeze(-1)                                # 父系轴角增量
        act = torch.zeros(1, 7, device=dev)
        act[:, 3:6] = rot_vec.clamp(-0.3, 0.3)
        env.step(act)
    return ang.item() * 57.3
