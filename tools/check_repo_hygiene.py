r"""check_repo_hygiene.py —— 仓库卫生检查（可作 pre-commit / CI）。

扫描**将被提交的内容**，命中以下任一模式即 fail：

1. 机器绝对路径：``[A-Za-z]:\`` / ``/[A-Za-z]/`` 盘符（提交内禁止真实盘符）
2. 嵌套 git 仓库：仓库内出现 ``.git`` 目录（第三方禁止 git clone 进 third_party）
3. 疑似密钥：``ghp_`` / ``github_pat_`` / ``sk-`` / ``-----BEGIN`` 等

实现：优先用 ``git ls-files``（仅跟踪文件），未 init 时退化为遍历（跳过
``outputs/ .git __pycache__`` 与 gitignored 的 ``config.local.yaml``）。

运行::

    python tools/check_repo_hygiene.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_vla.core.encoding import ensure_utf8_output  # noqa: E402

ensure_utf8_output()

# 忽略的目录/文件（与 .gitignore 一致）
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "outputs", ".idea", ".vscode", ".venv", "venv"}
SKIP_FILES = {"config.local.yaml", "config.local.yaml.example"}

ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/[A-Za-z]/")
# 用字符串拼接构造，避免扫描器把自己的正则源码误判为密钥
SECRET_RE = re.compile(
    r"github_" + r"pat_|ghp_|gho_|sk-[A-Za-z0-9]|-----BEGIN (RSA |OPENSSH |)PRIVATE KEY"
)
NESTED_GIT_RE = re.compile(r"^third_party/.*/\.git$|\.git/$")

# 文档 / 模板文件允许出现示例绝对路径（README 标注"示例路径"，.example 是本地模板）
_DOC_SUFFIXES = {".md", ".rst", ".txt"}
_TEMPLATE_SUFFIXES = {".example", ".template"}


def _is_code_file(path: Path) -> bool:
    """代码/配置文件才受"无绝对路径"约束；文档与模板不受。"""
    return path.suffix not in _DOC_SUFFIXES and not any(
        str(path).endswith(s) for s in _TEMPLATE_SUFFIXES
    )


def _tracked_files() -> list[Path]:
    """优先用 git ls-files；未 init 时退化为遍历。"""
    result = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return [Path(line) for line in result.stdout.split("\0") if line]
    # 退化：遍历（不尊重 .gitignore 的精确规则，按 SKIP_* 粗过滤）
    files: list[Path] = []
    for path in sorted(_ROOT.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.name in SKIP_FILES or path.suffix in (".pyc",):
            continue
        files.append(path)
    return files


def main() -> int:
    print(f"扫描仓库：{_ROOT}")
    files = _tracked_files()
    errors: list[tuple[Path, str, str]] = []

    for path in files:
        # 扫描器自身排除（其源码含正则模式；真实密钥绝不进仓库）
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # 1. 机器绝对路径（仅代码/配置文件；文档/模板允许示例路径）
        if _is_code_file(path):
            m = ABSOLUTE_PATH_RE.search(text)
            if m:
                errors.append((path, "机器绝对路径", m.group(0)))

        # 2. 疑似密钥（全类型检查）
        m = SECRET_RE.search(text)
        if m:
            errors.append((path, "疑似密钥", m.group(0)[:16] + "…"))

        # 3. 嵌套 .git 引用（文本声明）
        if NESTED_GIT_RE.search(text):
            errors.append((path, "嵌套 .git 引用", NESTED_GIT_RE.search(text).group(0)))

    # 检出嵌套 .git 目录（真实存在于仓库内）
    for git_dir in _ROOT.rglob(".git"):
        rel = git_dir.relative_to(_ROOT)
        if git_dir != _ROOT / ".git":  # 仓库自身的 .git 不算
            errors.append((git_dir, "嵌套 .git 目录", str(rel)))

    if errors:
        print(f"\n发现 {len(errors)} 个卫生问题：")
        for path, kind, detail in sorted(errors, key=lambda e: str(e[0])):
            print(f"  ✗ {path.relative_to(_ROOT) if path.is_relative_to(_ROOT) else path}: {kind} {detail}")
        print("\n请修复后再提交：机器路径移入 gitignored 的 config.local.yaml；密钥立即撤销；")
        print("第三方代码禁止 git clone 进仓库（见 third_party/README.md）。")
        return 1

    print(f"✓ 卫生检查通过：{len(files)} 个文件，无绝对路径 / 密钥 / 嵌套仓库")
    return 0


if __name__ == "__main__":
    sys.exit(main())
