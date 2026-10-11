"""Compatibility entry point for ``analysis.scenario_mix.summarize_scenario_mix``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.scenario_mix.summarize_scenario_mix import (  # noqa: E402,F401
    GROUP_ORDER,
    period_order,
    KEY_METRICS,
    load,
    ordered_groups,
    aggregate,
    print_table,
    EVENTS,
    real_price,
    plots,
    main,
)


if __name__ == "__main__":
    main()
