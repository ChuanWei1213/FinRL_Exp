"""Compatibility entry point for ``python -m finrl.experiments.evaluate_scenario_mix_checkpoints``."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep direct execution from a source checkout working without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments.evaluate_scenario_mix_checkpoints import (  # noqa: E402,F401
    TICKERS,
    eval_run,
    main,
)


if __name__ == "__main__":
    main()
