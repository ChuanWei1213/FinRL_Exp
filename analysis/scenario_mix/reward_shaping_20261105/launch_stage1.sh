#!/usr/bin/env bash
# Stage 1 of the reward-shaping study: screen 6 training rewards on two datasets.
#
#   datasets : PR30_B50 (joint-condition 70% full + 30% PR30) and V3_B50 (legacy fixed
#              crash scenarios, as in v3_nocost); both p = 0.5
#   rewards  : base / lin / sq / cube / exp / tail (see REWARDS below)
#   seeds    : 0-4; training commission 0, validation/test commission 0.25%
#   prices   : prices_2004_2025.csv for every run (one price file for the whole batch)
#
# The loss shapes are normalised to give the same penalty (2 reward units) for a 2% daily
# loss, about the 90th percentile of loss days in training: below 2% the convex shapes
# penalise less than linear, above 2% more.
#
# Usage (from the FinRL_Exp root):
#   bash analysis/scenario_mix/reward_shaping_20261105/launch_stage1.sh INPUT_DIR
#   INPUT_DIR holds full70_tail30_pr30.npz and legacy_fixed_scenarios.npz
set -euo pipefail

INPUTS=$(realpath "${1:?usage: launch_stage1.sh INPUT_DIR}")
PRICES=../TimeDiff/datasets/dow10/prices_2004_2025.csv
OUT=results/scenario_mix/reward_shaping_stage1_20261105
PARALLEL_BATCHES=3   # x WORKERS = training processes at once
WORKERS=5

REWARDS="base|
lin|--loss-weight 1
sq|--loss-weight 0.5 --loss-power 2
cube|--loss-weight 0.25 --loss-power 3 --loss-cap 30
exp|--loss-weight 0.582 --loss-exp-rate 0.5
tail|--tail-weight 2 --tail-threshold -0.05"

mkdir -p "$OUT"
echo "$REWARDS" > "$OUT/rewards.txt"

run_one() {
  local name=${1%%|*} extra=${1#*|}
  # shellcheck disable=SC2086  # $extra is a deliberate word list of CLI options
  python -m finrl.experiments.train_scenario_mix_batch \
    --prices "$PRICES" \
    --dataset PR30_B50 "$INPUTS/full70_tail30_pr30.npz" \
    --dataset V3_B50 "$INPUTS/legacy_fixed_scenarios.npz" \
    --p 0.5 --seeds 0 1 2 3 4 --workers "$WORKERS" \
    --timesteps 500000 --eval-freq 25000 \
    --commission 0.0025 --train-commission 0 \
    $extra --out "$OUT/$name" > "$OUT/$name.batch.log" 2>&1
  echo "finished $name ($(date +%H:%M))"
}
export -f run_one
export PRICES INPUTS OUT WORKERS

echo "$REWARDS" | xargs -P "$PARALLEL_BATCHES" -d '\n' -I{} bash -c 'run_one "$1"' _ {}
