"""轻量注册表：按名查表，根治旧工程"兄弟脚本互相 import"。

三类注册表：任务（sim/task）、策略（policy）、奖励构造（reward）。
用装饰器注册，用 :func:`get` / :meth:`Registry.get` 查表，查不到抛类型化异常。
"""

from __future__ import annotations

from typing import Any, Callable

from .errors import PolicyNotFoundError, TaskNotFoundError


class Registry:
    """通用按名注册表。"""

    def __init__(self, kind: str, not_found_error: type[Exception]):
        self._kind = kind
        self._error = not_found_error
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Callable], Callable]:
        """装饰器：把可调用对象按 name（默认取 ``__name__``）登记。"""

        def decorator(fn: Callable) -> Callable:
            key = name or fn.__name__
            if key in self._items:
                raise ValueError(f"{self._kind} 重复注册：{key!r}")
            self._items[key] = fn
            return fn

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise self._error(
                f"{self._kind} 未注册：{key!r}。可用：{sorted(self._items) or '（空）'}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)


# 全局注册表
TASKS = Registry("仿真任务", TaskNotFoundError)
POLICIES = Registry("策略后端", PolicyNotFoundError)
REWARD_BUILDERS = Registry("奖励构造器", TaskNotFoundError)


def register_task(name: str | None = None) -> Callable[[Callable], Callable]:
    return TASKS.register(name)


def register_policy(name: str | None = None) -> Callable[[Callable], Callable]:
    return POLICIES.register(name)


def register_reward_builder(name: str | None = None) -> Callable[[Callable], Callable]:
    return REWARD_BUILDERS.register(name)
