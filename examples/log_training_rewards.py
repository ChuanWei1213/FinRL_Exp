"""Compatibility entry point for ``python -m finrl.experiments.train_scenario_mix_logged``."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep direct execution from a source checkout working without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments.train_scenario_mix_logged import (  # noqa: E402,F401
    SourceInfo,
    TICKERS,
    main,
    make_ppo,
    validation_score,
)


if __name__ == "__main__":
    main()
