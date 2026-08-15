"""配置加载器：单一配置源 + 三层深合并 + 机器无关校验。

设计原则
--------
1. **唯一配置源**：所有配置写在 ``config/default.yaml``，代码里不维护平行的默认常量
   字典（旧工程 ``_DEFAULTS.num_views=1`` 与 ``yaml=3`` 漂移的根因就是配置有两份）。
2. **机器无关**：``default.yaml`` 中 ``paths`` 全部用 ``<local>/...`` 占位符，不出现
   真实盘符。真实路径只写在 gitignored 的 ``config.local.yaml``。
3. **三层覆盖**（优先级从低到高）：
   - ``config/default.yaml``（提交，兜底）
   - 覆盖来源：① ``$STAGE_VLA_CONFIG`` 指向的 yaml；② 仓库根 ``config.local.yaml``；
     ③ 环境变量逐键覆盖 ``STAGE_VLA_<SECTION>_<KEY>=<value>``。
4. **大声报错而非静默指向不存在的盘**：结构校验在加载时执行；需要真实路径的调用方
   用 :meth:`Settings.require_path` 获取，缺本地配置时抛出带指引的 ``ConfigError``。
5. **网络修复**：加载时把 S3 资产主机塞进 ``NO_PROXY``，规避国内代理把资产下载卡死
   （旧工程验证过的坑）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

# 仓库根（由本文件位置推导，代码零硬编码绝对路径）
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_DIR / "config.local.yaml"

# 校验：default.yaml 中必须存在的顶层键
_REQUIRED_TOP_LEVEL = ("project", "paths", "task", "stages", "thresholds", "reward_weights", "ppo", "lightweight", "rdt_normalization", "rdt_inference", "sim", "deploy")
_REQUIRED_THRESHOLDS = ("approach_dist", "grasp_reward_thresh", "lift_height", "place_align_dist")

# S3 资产主机（Isaac 资产下载用），塞进 NO_PROXY 防国内代理卡死
_S3_HOSTS = (
    "amazonaws.com",
    "omniverse-content-production.s3.us-west-2.amazonaws.com",
    "omniverse-content-production.s3-us-west-2.amazonaws.com",
)

# 环境变量逐键覆盖前缀（STAGE_VLA_CONFIG 是配置文件路径，其余按节_键拆分）
_ENV_PREFIX = "STAGE_VLA_"
_ENV_CONFIG_PATH = "STAGE_VLA_CONFIG"


def _apply_no_proxy_fix() -> None:
    """把 S3 资产主机加进 NO_PROXY，规避国内代理把资产下载卡死。"""
    current = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    missing = [h for h in _S3_HOSTS if h not in current]
    if missing:
        merged = ",".join([current] + missing).strip(",")
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：``override`` 的键覆盖 ``base``，字典向下合并，其余直接替换。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_env_value(raw: str) -> Any:
    """把环境变量字符串转成合适的 Python 类型（bool/int/float/list 等）。"""
    raw = raw.strip()
    lower = raw.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            pass
    return raw


def _env_key_overrides() -> dict:
    """解析 ``STAGE_VLA_<SECTION>_<KEY>`` 形式的环境变量为嵌套字典。"""
    merged: dict = {}
    prefix_len = len(_ENV_PREFIX)
    for name, value in os.environ.items():
        if not name.startswith(_ENV_PREFIX) or name == _ENV_CONFIG_PATH:
            continue
        parts = name[prefix_len:].split("_")
        node = merged
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_env_value(value)
    return merged


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件解析失败：{path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return data


def resolve_config_paths() -> tuple[Path, Path]:
    """返回 (default_path, local_path)。local 按优先级：环境变量 > 仓库根。"""
    env_path = os.environ.get(_ENV_CONFIG_PATH)
    local = Path(env_path) if env_path else LOCAL_CONFIG_PATH
    return DEFAULT_CONFIG_PATH, local


def load_settings(
    default_path: Path | None = None,
    local_path: Path | None = None,
    *,
    allow_placeholder_paths: bool = True,
) -> "Settings":
    """加载并合并配置，返回只读 :class:`Settings`。

    Parameters
    ----------
    default_path, local_path : 显式指定配置文件路径（测试用）。
    allow_placeholder_paths : 若 False，``paths`` 中残留 ``<...>`` 占位符即抛错。
    """
    default_path, local_path = default_path or DEFAULT_CONFIG_PATH, local_path or LOCAL_CONFIG_PATH

    _apply_no_proxy_fix()

    cfg = _load_yaml(default_path)
    if local_path.is_file():
        cfg = deep_merge(cfg, _load_yaml(local_path))
    cfg = deep_merge(cfg, _env_key_overrides())

    settings = Settings(cfg)
    settings._validate()

    if not allow_placeholder_paths:
        for key, path in settings.paths.items():
            settings.require_path(key)
    return settings


def require_paths(settings: "Settings", *keys: str) -> None:
    """一次性获取多个真实路径，缺任何一个本地配置就大声报错。"""
    for key in keys:
        settings.require_path(key)


class Settings:
    """只读配置对象。

    用法::

        settings = load_settings()
        settings.ppo["num_envs"]          # 字典节
        settings.paths["isaaclab"]        # Path 对象
        settings.require_path("isaaclab") # 缺本地配置时抛 ConfigError
    """

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)
        object.__setattr__(
            self,
            "_paths",
            {k: Path(v) for k, v in data.get("paths", {}).items()},
        )

    @property
    def raw(self) -> dict:
        return self._data

    @property
    def paths(self) -> dict[str, Path]:
        return self._paths

    def require_path(self, key: str) -> Path:
        """返回真实路径；若仍是 ``<...>`` 占位或为空，抛带指引的 ConfigError。"""
        path = self._paths.get(key)
        if path is None:
            raise ConfigError(f"配置缺少 paths.{key}。请在 config.local.yaml 配置（见 config/config.local.yaml.example）。")
        text = str(path)
        if "<" in text or ">" in text or not text:
            raise ConfigError(
                f"paths.{key} 仍是占位符 {text!r}。真实路径请写在 gitignored 的 "
                f"config.local.yaml（或设环境变量 STAGE_VLA_PATHS_{key.upper()}）。"
            )
        return path

    def _validate(self) -> None:
        """结构完整性校验：键存在性 / stages 与 reward_weights 对齐 / 阈值完整。"""
        for key in _REQUIRED_TOP_LEVEL:
            if key not in self._data:
                raise ConfigError(f"default.yaml 缺少顶层键 {key!r}，请勿删除。")

        stages: list = self._data["stages"]
        if not isinstance(stages, list) or not stages:
            raise ConfigError("stages 必须是非空列表（子阶段序列）。")

        weights: dict = self._data["reward_weights"]
        for stage in stages:
            if stage not in weights:
                raise ConfigError(f"reward_weights 缺少阶段 {stage!r}（键需与 stages 一一对应）。")
        for key in ("action_penalty", "progress_shaping"):
            if key not in weights:
                raise ConfigError(f"reward_weights 缺少 {key!r}。")

        for key in _REQUIRED_THRESHOLDS:
            if key not in self._data["thresholds"]:
                raise ConfigError(f"thresholds 缺少 {key!r}。")

        norm: dict = self._data["rdt_normalization"]
        for key in ("state_min", "state_max", "action_min", "action_max"):
            if key not in norm or len(norm[key]) != 8:
                raise ConfigError(f"rdt_normalization.{key} 必须是 8 维列表（7 关节 + 夹爪）。")

    def __getattr__(self, name: str) -> Any:
        """按顶层节访问：settings.task / settings.stages / settings.ppo ..."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在顶层键 {name!r}") from exc


# 训练必需路径（缺失即判定机器未就绪）；其余 VLA 模型路径为 M2/M3 可选
REQUIRED_PATHS = ("isaaclab", "isaac_sim")


def check_machine(verbose: bool = True) -> dict:
    """打印解析后的配置与本机路径存在性，返回报告 dict（供 tools/check_machine.py 调用）。

    Returns:
        {"settings": dict, "paths": {key: {path, exists}}, "missing": [keys],
         "missing_required": [keys]}  缺失路径分必需/可选两类
    """
    settings = load_settings()
    report: dict[str, Any] = {
        "settings": settings.raw,
        "paths": {},
        "missing": [],
        "missing_required": [],
    }
    if verbose:
        print("=" * 60)
        print("解析后的配置（已合并 local + 环境变量）")
        print("=" * 60)
        for key, path in settings.paths.items():
            exists = path.exists()
            report["paths"][key] = {"path": str(path), "exists": exists}
            if not exists:
                report["missing"].append(key)
                if key in REQUIRED_PATHS:
                    report["missing_required"].append(key)
            tag = "✓" if exists else ("✗ [必需]" if key in REQUIRED_PATHS else "✗ [可选]")
            print(f"  paths.{key:<16} {tag} {path}")
    return report
