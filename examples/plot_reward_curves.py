"""Compatibility entry point for ``analysis.scenario_mix.plot_reward_curves``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.scenario_mix.plot_reward_curves import (  # noqa: E402,F401
    GROUPS,
    SEED_COLORS,
    PICKS,
    ROWS,
    load,
    main,
)


if __name__ == "__main__":
    main()
