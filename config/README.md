# 配置体系

## 单一配置源

所有配置写在 `config/default.yaml`（提交，**机器无关**）。代码中不维护平行的默认常量
字典——旧工程 `_DEFAULTS.num_views=1` 与 `config.yaml=3` 漂移的根因就是"配置有两份"。

## 三层覆盖（优先级从低到高）

| 优先级 | 来源 | 说明 |
|---|---|---|
| 1（兜底） | `config/default.yaml` | 提交，全部路径为 `<local>/...` 占位符 |
| 2 | `$STAGE_VLA_CONFIG` 指向的 yaml | 环境变量显式指定覆盖文件 |
| 3 | 仓库根 `config.local.yaml` | gitignored，本机真实路径 |
| 4 | 环境变量 `STAGE_VLA_<SECTION>_<KEY>` | 逐键覆盖，如 `STAGE_VLA_PATHS_ISAACLAB=D:\...` |

> 注：`STAGE_VLA_CONFIG` 特指"覆盖文件路径"，不会被当作键覆盖。

## 真实路径约定

- 提交文件**零机器绝对路径**（`E:\` / `D:\` / `C:\Users\...`）。
- 本机真实路径只写进 `config.local.yaml`（已 gitignore）。
- `tools/check_repo_hygiene.py` 会在 CI 里扫描并拒绝提交中出现盘符绝对路径 / 嵌套 `.git` / 密钥。

## 使用示例

```py
from stage_vla.core.config import load_settings

settings = load_settings()
settings.ppo["num_envs"]             # 128
settings.stages                       # ['approach','grasp','lift','move','stack']
isaaclab = settings.require_path("isaaclab")  # 缺本地配置时抛 ConfigError
```

## 首次使用

```bat
copy config\config.local.yaml.example config.local.yaml
REM 编辑 config.local.yaml 填入本机路径
```
