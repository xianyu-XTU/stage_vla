# third_party 第三方代码政策

本项目**禁止以嵌套 git 仓库（git submodule / git clone 进本目录）的方式引用第三方代码**。
旧工程 `E:\stage_vla\third_party\rdt_repo` 是一个完整的独立 git 仓库，曾导致父仓库出现嵌套 .git 的卫生问题。

## 允许的引入方式

1. **pip 包安装**（首选）：OpenVLA / transformers / peft / bitsandbytes 等一律通过 pip 安装，路径记录在 `pyproject.toml` 或 `config/local.yaml`。
2. **拷贝 + 保留 LICENSE 头**：如确需修改少量第三方文件（如 RDT 推理代码），按"拷贝文件 + 保留原 LICENSE 与版权头"的方式纳入，并在本 README 的引入清单中登记来源与版本。

## 引入登记（随引入更新）

| 组件 | 来源 | 版本/commit | 引入方式 | 用途 |
|---|---|---|---|---|
| （待引入） | | | | |
