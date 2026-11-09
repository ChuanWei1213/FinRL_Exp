#!/usr/bin/env bash
# Run the REAL_ONLY / REAL_REPLAY controls, then evaluate all 120 stage-1 runs.
# Start it after launch_stage1.sh has finished (no process wait: `pgrep -f` on the script
# name also matches the shell that launched this one, so such a wait never ends).
# Usage (from the FinRL_Exp root): bash .../run_stage1_remaining.sh INPUT_DIR
set -euo pipefail
INPUTS=${1:?usage: run_stage1_remaining.sh INPUT_DIR}
HERE=analysis/scenario_mix/reward_shaping_20261105
bash "$HERE/launch_stage1_controls.sh" "$INPUTS"
echo "controls done $(date +%H:%M)"
python -m analysis.scenario_mix.reward_shaping_20261105.evaluate \
  --root results/scenario_mix/reward_shaping_stage1_20261105 \
  --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv --workers 15
echo "EVAL DONE $(date +%H:%M)"
