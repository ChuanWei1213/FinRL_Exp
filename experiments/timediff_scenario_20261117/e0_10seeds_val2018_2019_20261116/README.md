Three E0 conditions, lambda=0.5, PPO seeds1..10. Train2005-2017, validation targets2018-2019, actual event first21 test.

Session: timediff-e0-10seeds-20261116

Attach:
```bash
tmux -S /mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/tmux.sock attach -t timediff-e0-10seeds-20261116
```

Resume from /home/cwei1213/FinRL_Exp:
```bash
/opt/anaconda3/envs/finrl_exp/bin/python -u -m finrl.experiments.timediff_e0_multiseed run --out /mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116 --workers 8 --threads 2
```

Progress: status.json, run.log, jobs/*/status.json. Final outputs: REPORT.md, charts.pdf, test_summary.csv, paired_comparisons.csv, selected_checkpoints.csv, audit.json.
