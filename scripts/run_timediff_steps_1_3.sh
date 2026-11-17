#!/bin/bash
set -euo pipefail
cd /home/cwei1213/FinRL_Exp
pilot_out=/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/home/cwei1213/FinRL_Exp/results/timediff_runtime_tmp
mkdir -p "$TMPDIR" "$pilot_out"
exec >> "$pilot_out/master.log" 2>&1
/opt/anaconda3/envs/finrl_exp/bin/python -u scripts/verify_timediff_resume.py --out "$pilot_out"
exec /opt/anaconda3/envs/finrl_exp/bin/python -u -m finrl.experiments.timediff_scenario_pilot run --out "$pilot_out" --workers 2 --resume
