#!/usr/bin/env bash
# Stage 1 controls: the same 6 rewards x 5 seeds on REAL_ONLY and REAL_REPLAY_PR30_B50.
#
# Companion of launch_stage1.sh (PR30_B50, V3_B50); every other setting is identical:
# prices_2004_2025.csv, seeds 0-4, 500k nominal steps, training commission 0,
# validation commission 0.25%. Output batches sit next to the stage-1 ones:
#   <reward>__replay    REAL_REPLAY_PR30_B50, p = 0.5 (paired real futures of the PR30 draws)
#   <reward>__realonly  REAL_ONLY, p = 0 (the dataset file is required by the CLI, never read)
#
# Usage (from the FinRL_Exp root):
#   bash analysis/scenario_mix/reward_shaping_20261105/launch_stage1_controls.sh INPUT_DIR
set -euo pipefail

INPUTS=$(realpath "${1:?usage: launch_stage1_controls.sh INPUT_DIR}")
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

# one line per batch: reward|suffix|dataset name|npz|p|extra options
JOBS=$(echo "$REWARDS" | while IFS='|' read -r name extra; do
  echo "$name|replay|REAL_REPLAY_PR30_B50|real_replay_pr30.npz|0.5|$extra"
  echo "$name|realonly|REAL_ONLY|full70_tail30_pr30.npz|0|$extra"
done)

run_one() {
  local name suffix group npz p extra
  IFS='|' read -r name suffix group npz p extra <<< "$1"
  # shellcheck disable=SC2086  # $extra is a deliberate word list of CLI options
  python -m finrl.experiments.train_scenario_mix_batch \
    --prices "$PRICES" \
    --dataset "$group" "$INPUTS/$npz" \
    --p "$p" --seeds 0 1 2 3 4 --workers "$WORKERS" \
    --timesteps 500000 --eval-freq 25000 \
    --commission 0.0025 --train-commission 0 \
    $extra --out "$OUT/${name}__$suffix" > "$OUT/${name}__$suffix.batch.log" 2>&1
  echo "finished ${name}__$suffix ($(date +%H:%M))"
}
export -f run_one
export PRICES INPUTS OUT WORKERS

echo "$JOBS" | xargs -P "$PARALLEL_BATCHES" -d '\n' -I{} bash -c 'run_one "$1"' _ {}
