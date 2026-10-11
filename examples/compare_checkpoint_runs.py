"""Compatibility entry point for ``analysis.scenario_mix.compare_checkpoint_runs``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.scenario_mix.compare_checkpoint_runs import (  # noqa: E402,F401
    GROUPS,
    COLORS,
    ROWS,
    TABLE_PERIODS,
    TABLE_METRICS,
    load,
    main,
)


if __name__ == "__main__":
    main()
