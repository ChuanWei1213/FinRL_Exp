Random k=0..21 rerun; seed=0, TimeDiff lambda=0.5.

Session: timediff-random-k-20261116

Attach:
```bash
tmux -S /mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/tmux.sock attach -t timediff-random-k-20261116
```

Resume automatically from completed jobs and PPO checkpoints:
```bash
/opt/anaconda3/envs/finrl_exp/bin/python -u -m finrl.experiments.timediff_random_k run --out /mnt/sda/cwei1213/timediff_finrl/random_k_20261116 --workers 8
```

Run from /home/cwei1213/FinRL_Exp. Logs: run.log, jobs/*/training.log. Status: status.json. Final report: REPORT.md.
