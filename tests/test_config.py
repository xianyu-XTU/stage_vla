"""配置加载测试：卫生约束 / 键对齐 / RDT 归一化硬约束。"""

from __future__ import annotations

import re

import pytest

from stage_vla.core.config import DEFAULT_CONFIG_PATH, ConfigError, deep_merge, load_settings


def test_default_config_no_absolute_paths():
    """提交的 default.yaml 不得包含盘符绝对路径（机器无关约束）。"""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert not re.findall(r"[A-Za-z]:\\", text), "default.yaml 出现盘符绝对路径"


def test_config_loads_and_merges(settings):
    assert settings.stages == ["approach", "grasp", "lift", "move", "stack"]
    assert settings.ppo["num_envs"] == 128
    for key in (
        "isaaclab", "isaac_sim", "openvla_root", "openvla_model",
        "rdt_model", "siglip_model", "t5_model", "lang_embed", "rdt_repo",
    ):
        assert key in settings.paths, f"paths 缺 {key}"


def test_placeholder_paths_when_no_local(tmp_path):
    """无本地配置时 paths 保留占位符，且 require_path 大声报错。"""
    default = DEFAULT_CONFIG_PATH
    bad = tmp_path / "no_local.yaml"  # 不存在的本地文件
    settings = load_settings(default_path=default, local_path=bad)
    assert "<local>" in str(settings.paths["isaaclab"])
    with pytest.raises(ConfigError, match="占位符"):
        settings.require_path("isaaclab")


def test_reward_weights_aligned_with_stages(settings):
    for stage in settings.stages:
        assert stage in settings.reward_weights, f"reward_weights 缺 {stage}"
    for key in ("action_penalty", "progress_shaping"):
        assert key in settings.reward_weights


def test_thresholds_complete(settings):
    for key in ("approach_dist", "grasp_reward_thresh", "lift_height", "place_align_dist"):
        assert key in settings.thresholds


def test_rdt_normalization_8dim_hard_constraint(settings):
    """RDT 归一化 8 维（7 关节 + 夹爪）是硬约束，防止配置漂移。"""
    for key in ("state_min", "state_max", "action_min", "action_max"):
        vals = settings.rdt_normalization[key]
        assert len(vals) == 8, f"rdt_normalization.{key} 必须 8 维，当前 {len(vals)}"


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3, "z": 4}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1}


def test_invalid_config_raises(tmp_path):
    """缺 stages 键时加载必须大声报错。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: {name: x}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(default_path=bad, local_path=tmp_path / "none.yaml")
