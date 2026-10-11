"""Compatibility entry point for ``analysis.scenario_mix.compare_selection_metrics``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.scenario_mix.compare_selection_metrics import (  # noqa: E402,F401
    GROUPS,
    METRICS,
    FIG_METRICS,
    FIG_PERIODS,
    COLORS,
    parse_runs,
    valid_curves,
    selected,
    figure,
    main,
)


if __name__ == "__main__":
    main()
