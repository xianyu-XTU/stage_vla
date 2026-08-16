"""指令分词器测试：不同指令不同嵌入、形状、池化正确、参数小。"""

from __future__ import annotations

import pytest
import torch

from stage_vla.vla_light.instruction_tokenizer import InstructionTokenizer


@pytest.fixture(scope="module")
def tok() -> InstructionTokenizer:
    # 用真实 T5 tokenizer（本机只读引用）；eval 关闭 dropout 保证确定性
    t = InstructionTokenizer(tokenizer_path=r"D:\vla_models\t5-xxl", hidden=128, num_layers=2, num_heads=4)
    t.eval()
    return t


def test_shape_and_dtype(tok):
    out = tok("pick up the red cube")
    assert out.shape == (1, 128)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


def test_different_instructions_differ(tok):
    a = tok("pick up the red cube and place it on the blue cube")
    b = tok("push the green cube to the left")
    c = tok("pick up the red cube and place it on the blue cube")  # 相同指令
    # 不同指令嵌入应不同，相同指令嵌入应相同
    assert (a - b).norm().item() > 1e-2, "不同指令嵌入应不同"
    torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)


def test_batch(tok):
    out = tok(["grab the cube", "stack the red cube on the blue"])
    assert out.shape == (2, 128)


def test_size_is_small(tok):
    """指令分词器应远小于 7B（对比：Llama-2 7B ≈ 6.7e9 参数）。"""
    n = tok.param_count()
    assert n < 100_000_000, f"指令分词器应 <100M，实际 {n/1e6:.1f}M"
    assert n / 6.7e9 < 0.02, "应 < Llama-2 7B 的 2%"


def test_padding_handling(tok):
    """不同长度指令经 padding 后嵌入应有限且不同。"""
    short = tok("go")
    long = tok("pick up the red cube and then carefully place it on top of the blue cube without knocking it over")
    assert short.shape == (1, 128)
    assert long.shape == (1, 128)
    assert bool(torch.isfinite(long).all())
    assert (short - long).norm().item() > 1e-2
