# 仿真与数据集对接计划

## 当前（M0~M3）

只使用 **Isaac Lab / Isaac Sim**（`envs/cfg_surgery.py` + `rl/runner.py`），通过
`config/task` 的 `id_state`（IK 相对状态版）等任务驱动。

## M4 预留

| 目标 | 文件 | 说明 |
|---|---|---|
| 仿真网关抽象 | `envs/sim_gateway.py` | `SimGateway` ABC：make_env / step / get_obs / get_stage_signals |
| Gazebo 网关 | `envs/sim_gateway.py`（GazeboGateway） | 申请书承诺，M4 落地 |
| ManiSkill 3 适配 | `data/maniskill.py`（预留） | 标准化评估基准（旧工程零对接） |
| SimpleReNV 适配 | `data/simplerenv.py`（预留） | 真实到虚拟基准 |

## 评估验收（M4）

同一 StARe-PPO 权重在 Isaac Lab 与 ManiSkill 3 测试集上评估，产出成功率对比表
（写入 `outputs/reports/`）。
