"""Disable the unused legacy report dependency for this PPO invocation only."""
import os
import sys

if os.environ.get("FINRL_SCENARIO_DISABLE_QUANTSTATS") == "1":
    sys.modules.setdefault("quantstats", None)
