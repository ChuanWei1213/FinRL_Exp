"""Compatibility entry point for ``python -m finrl.experiments.evaluate_scenario_mix``."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep direct execution from a source checkout working without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments.evaluate_scenario_mix import (  # noqa: E402,F401
    TICKERS,
    main,
)


if __name__ == "__main__":
    main()
