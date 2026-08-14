from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Any


def _disable_installed_gym_notice() -> None:
    """Hide Gym's import-time maintenance notice without muting stderr."""
    try:
        import gym_notices.notices as gym_notices

        gym_notices.notices.pop(version("gym"), None)
    except (ImportError, PackageNotFoundError):
        pass


_disable_installed_gym_notice()


__all__ = ("test", "trade", "train")

_LAZY_ENTRY_POINTS = {
    "test": "finrl.test",
    "trade": "finrl.trade",
    "train": "finrl.train",
}


def __getattr__(name: str) -> Any:
    """Load top-level entry points only when they are requested.

    In particular, importing ``finrl`` or an experiment module should not
    require the optional Alpaca paper-trading dependencies pulled in by
    ``finrl.trade``.
    """
    module_name = _LAZY_ENTRY_POINTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_ENTRY_POINTS))
