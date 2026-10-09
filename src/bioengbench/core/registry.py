"""Plugin registry: every module in `bioengbench.tasks` that defines a
`@register`-decorated TaskPlugin is discovered automatically."""

from __future__ import annotations

import importlib
import pkgutil

import bioengbench.tasks as _tasks_package
from bioengbench.core.task import TaskPlugin

_REGISTRY: dict[str, type[TaskPlugin]] = {}


def register(cls: type[TaskPlugin]) -> type[TaskPlugin]:
    if cls.task_id in _REGISTRY and _REGISTRY[cls.task_id] is not cls:
        raise ValueError(f"duplicate task id {cls.task_id!r}")
    _REGISTRY[cls.task_id] = cls
    return cls


def _discover() -> None:
    for info in pkgutil.iter_modules(_tasks_package.__path__):
        importlib.import_module(f"{_tasks_package.__name__}.{info.name}")


def task_ids() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def get(task_id: str) -> TaskPlugin:
    _discover()
    try:
        return _REGISTRY[task_id]()
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(_REGISTRY)}") from None
