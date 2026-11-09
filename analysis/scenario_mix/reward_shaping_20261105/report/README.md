# Reward-shaping stage 1: report files

Figures and tables behind the Notion report "Reward 改造第一階段：懲罰虧損與尾部".
All inputs come from `results/scenario_mix/reward_shaping_stage1_20261105/`
(`selected_metrics.csv`, `checkpoint_metrics.csv`, `daily/`), written by `../evaluate.py`.

Run from the FinRL_Exp root with `R=results/scenario_mix/reward_shaping_stage1_20261105`
and `D=analysis/scenario_mix/reward_shaping_20261105/report`:

    python $D/summarize.py $R                                    # per dataset x reward tables (stdout)
    PYTHONPATH=. python $D/timing_vs_static_mix.py $R $D/timing_vs_static_mix.csv
    PYTHONPATH=. python $D/event_segments.py $R $D                # event tables (backtests checkpoints)
    PYTHONPATH=. python $D/event_figures.py $D                       # 04-05 event PNGs
    PYTHONPATH=. python $D/figures.py $R $D                      # 01-03 PNGs; needs timing_vs_static_mix.csv

- `timing_vs_static_mix`: each selected checkpoint against a constant-mix portfolio
  (its own mean stock weight in the 10 stocks equal-weight, rest cash, daily rebalanced,
  no fees). `timing_PR10` is the mean 21-day excess over that portfolio in PR10 down windows.
- `event_segments`: drawdown / 252-day recovery / full-event metrics for the 2020 COVID and 2022
  bear-market events, with the event dates and table format of the "走弱行情的保護與反彈代價｜
  Scenario Mix 2020／2022" report (each event is a fresh backtest from its activation date).
  Writes `event_metrics_by_seed.csv`, `event_summary.csv`, `buy_and_hold_events.csv` and
  `event_daily.csv.gz`; its buy & hold numbers match that report exactly.
- `figures.py` uses the Noto Sans CJK font at `/usr/share/fonts/opentype/noto/`.

Checkpoints (`ckpt/step_*.zip`) are not tracked in git; they stay on the training server.
