"""train_primitives.py —— 训练基础动作策略的两个可学习组件。

阶段 1（plan，无需 Isaac）：**计划解码器** —— 用语义计划（semantic_plan）当监督，
训练 instruction_tokenizer + plan_decoder 学会"指令 → 基础动作序列"。

阶段 2（grounder，需 Isaac）：**视觉引导器** —— 仿真 rollout 采集 (图像, 当前阶段) 对，
用 StageDetector 当监督，训练 primitive_head 学会"看到什么状态 → 哪个基础动作"。

用法::

    python scripts\\train_primitives.py --phase plan      # 只训计划解码器（快，无需 Isaac）
    python tools\\run_isaaclab.py scripts\\train_primitives.py --phase grounder --iters 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from stage_vla.core.config import load_settings  # noqa: E402
from stage_vla.vla_light.planner import STOP_IDX, semantic_plan  # noqa: E402
from stage_vla.vla_light.primitives import PRIMITIVE_INDEX, PRIMITIVES  # noqa: E402
from stage_vla.vla_light.vision_policy import VisionPrimitivePolicy  # noqa: E402

PLAN_LEN = 5


# ============================================================================
# 指令模板生成（覆盖不同长度/语义的计划）
# ============================================================================
def generate_instructions() -> list[str]:
    colors = ["red", "blue", "green"]
    instrs: list[str] = []
    for c1 in colors:
        for c2 in colors:
            if c1 == c2:
                continue
            instrs += [
                f"pick up the {c1} cube and place it on the {c2} cube",
                f"grab the {c1} cube and put it on the {c2} cube",
                f"stack the {c1} cube on the {c2} cube",
                f"just pick up the {c1} cube",
                f"pick up the {c1} cube",
            ]
    return sorted(set(instrs))


def plan_label(instruction: str) -> torch.Tensor:
    """指令 → 目标计划索引（不足 PLAN_LEN 的位置填 STOP，让解码器学输出 STOP）。"""
    names = semantic_plan(instruction)
    target = [PRIMITIVE_INDEX[n] for n in names]
    target += [STOP_IDX] * (PLAN_LEN - len(target))
    return torch.tensor(target, dtype=torch.long)


# ============================================================================
# 阶段 1：计划解码器
# ============================================================================
def train_plan_decoder(model: VisionPrimitivePolicy, epochs: int = 60, lr: float = 1e-3, device: str = "cuda"):
    """用语义计划当监督，训练 instruction_tokenizer + plan_decoder。"""
    instrs = generate_instructions()
    print(f"[phase1] 训练计划解码器：{len(instrs)} 条指令 / {epochs} epochs")

    params = list(model.instruction_tokenizer.parameters()) + list(model.plan_decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()   # 所有位置都有有效标签（含 STOP）

    for epoch in range(epochs):
        total = 0.0
        for instr in instrs:
            target = plan_label(instr).to(device)
            emb = model.instruction_tokenizer(instr).to(device)
            logits = model.plan_decoder(emb)                          # [1, PLAN_LEN, n_prim]
            loss = loss_fn(logits[0], target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3d} | loss {total/len(instrs):.4f}")

    # 验证：随机抽几条指令
    print("  [验证] 计划解码：")
    model.eval()
    with torch.no_grad():
        for instr in instrs[::7]:
            emb = model.instruction_tokenizer(instr).to(device)
            pred = model.plan_decoder.decode(emb)[0]
            true = semantic_plan(instr)
            ok = "✓" if pred == true else "✗"
            print(f"    {ok} {instr[:42]:<44} 真值={true}  预测={pred}")
    model.train()


# ============================================================================
# 阶段 2：视觉引导器
# ============================================================================
def train_vision_grounder(model: VisionPrimitivePolicy, iters: int = 60, horizon: int = 24,
                          num_envs: int = 4, lr: float = 1e-3, cam_res: int = 96, device: str = "cuda"):
    """仿真 rollout 采集 (图像, 当前阶段)，训练 primitive_head 分类。

    注意：随机动作下阶段分布可能偏斜（多为 approach），报告逐阶段分布以便诚实评估。
    """
    from collections import Counter

    from stage_vla.envs.cfg_surgery import build_vision_env_cfg
    from stage_vla.rl.runner import create_raw_env
    from stage_vla.stages.rewards_isaac import detect_stage_from_env

    settings = load_settings()
    env_cfg = build_vision_env_cfg(settings, num_envs=num_envs, seed=42, cam_res=cam_res)
    env, sim_app = create_raw_env(settings.task["id_visuomotor"], env_cfg, headless=True, enable_cameras=True)

    instr = settings.task["desc"]
    opt = torch.optim.Adam(model.primitive_head.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    env_dev = env.unwrapped.device

    print(f"[phase2] 训练视觉引导器：{iters} iter × {horizon} 步 × {num_envs} env", flush=True)
    obs, _ = env.reset()
    for it in range(iters):
        total, correct, n = 0.0, 0, 0
        stage_counts = Counter()
        for _ in range(horizon):
            # 随机（限幅）动作让机械臂动起来，采集当前阶段的图像
            action = (torch.rand(env.action_space.shape, device=env_dev) * 2 - 1) * 0.3
            obs, rew, term, trunc, _ = env.step(action)
            stage = detect_stage_from_env(env.unwrapped)              # [N] 当前阶段（老师）
            stage_counts.update(stage.tolist())

            img = obs["policy"]["table_cam"]
            with torch.no_grad():
                emb = model.instruction_tokenizer(instr).to(device)
            fused = model._fuse(img, emb, 0)                          # 视觉+指令（step=0）
            logits = model.primitive_head(fused)                      # [N, n_prim]
            loss = loss_fn(logits, stage.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            correct += (logits.argmax(-1) == stage.to(device)).sum().item()
            n += num_envs
            # 重置终止的环境
            if bool((term | trunc).any()):
                obs, _ = env.reset()
        acc = correct / n
        if it % 10 == 0 or it == iters - 1:
            print(f"  iter {it:>3d} | loss {total/horizon:.4f} | acc {acc:.3f} | "
                  f"阶段分布 {dict(sorted(stage_counts.items()))}", flush=True)
    env.close()
    sim_app.close()


def train_vision_grounder_data(model: VisionPrimitivePolicy, epochs: int = 150, lr: float = 1e-3,
                               data_path: str = "outputs/primitive_data.pt", device: str = "cuda"):
    """用脚本化采集的均衡 (图像, 阶段) 数据训练 primitive_head（无需 Isaac）。

    视觉特征 + 指令嵌入一次预提取，随后快速分类训练；报告逐阶段准确率。
    """
    from collections import Counter

    data = torch.load(_ROOT / data_path, map_location="cpu", weights_only=False)
    imgs = data["images"]                    # [N, 96, 96, 3] uint8
    stages = torch.tensor(data["stages"], dtype=torch.long)
    dist = dict(sorted(Counter(stages.tolist()).items()))
    print(f"[phase2-data] 加载 {len(imgs)} 张，分布 {dist}", flush=True)

    instr = "pick up the red cube and place it on the blue cube"
    with torch.no_grad():
        feats = [model.vision(imgs[i:i + 8]) for i in range(0, len(imgs), 8)]
        feats = torch.cat(feats).to(device)                # [N, 4096]
        instr_emb = model.instruction_tokenizer(instr).to(device)   # [1, 512]
    N = feats.shape[0]
    step_ctx = model._step_onehot(0, N, feats.device)      # 常量上下文（分类靠视觉）
    fused = torch.cat([feats, instr_emb.expand(N, -1), step_ctx], dim=1)

    opt = torch.optim.Adam(model.primitive_head.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 8):
            idx = perm[i:i + 8]
            logits = model.primitive_head(fused[idx])
            loss = loss_fn(logits, stages[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = model.primitive_head(fused).argmax(-1)
                acc = (pred == stages.to(device)).float().mean().item()
                per_stage = [(s, ((pred == s) & (stages.to(device) == s)).float().mean().item())
                             for s in range(5)]
            print(f"  epoch {epoch:>3d} | loss {loss.item():.4f} | acc {acc:.3f} | 逐阶段 {per_stage}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练基础动作策略（计划解码器 + 视觉引导器）")
    parser.add_argument("--phase", choices=["plan", "grounder", "grounder_data", "all"], default="all")
    parser.add_argument("--plan_epochs", type=int, default=60)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--cam_res", type=int, default=96)
    parser.add_argument("--grounder_epochs", type=int, default=150)
    args = parser.parse_args()

    settings = load_settings()
    model = VisionPrimitivePolicy(settings)

    if args.phase in ("plan", "all"):
        train_plan_decoder(model, epochs=args.plan_epochs)
    if args.phase == "grounder":
        train_vision_grounder(model, iters=args.iters, num_envs=args.num_envs, cam_res=args.cam_res)
    if args.phase in ("grounder_data", "all"):
        train_vision_grounder_data(model, epochs=args.grounder_epochs)

    # 保存
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "outputs/primitives_model.pt")
    print(f"[save] 已保存 outputs/primitives_model.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
