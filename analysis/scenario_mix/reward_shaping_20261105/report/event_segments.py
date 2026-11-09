"""Drawdown / recovery / full-event metrics for the two stress events, in the format of the
"走弱行情的保護與反彈代價｜Scenario Mix 2020／2022" report.

Each event is a fresh deterministic backtest from its activation date (100,000, 1/11 in cash and
each stock, 0.25% TRF commission) with the checkpoint picked by 2018-2019 validation Sharpe.

    event      activation  peak        trough      trough + 252 trading days
    covid2020  2020-01-02  2020-02-12  2020-03-16  2021-03-16
    bear2022   2021-09-29  2021-12-29  2022-09-30  2023-10-03

Segments: down (peak close -> trough close), recovery (trough close -> +252 days), full (peak ->
end). Return = end/start - 1; MDD = min(value/cummax - 1); Sharpe = mean/std(ddof=1) of daily log
returns x sqrt(252); exposure = mean post-drift stock weight over the segment's trading days
(start day excluded). Buy & hold = 100% stocks, equal weight at activation.

    PYTHONPATH=. python event_segments.py RESULTS_ROOT OUT_DIR
writes event_metrics_by_seed.csv, event_summary.csv, event_daily.csv.gz and buy_and_hold_events.csv.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PRICES = Path("../TimeDiff/datasets/dow10/prices_2004_2025.csv")
COMMISSION = 0.0025
RECOVERY_DAYS = 252
EVENTS = {"covid2020": ("2020-01-02", "2020-02-12", "2020-03-16"),
          "bear2022": ("2021-09-29", "2021-12-29", "2022-09-30")}
METRICS = ["return", "mdd", "sharpe", "exposure"]


def event_dates(dates, activation, peak, trough):
    dates = pd.DatetimeIndex(dates)
    end = dates[dates.get_loc(pd.Timestamp(trough)) + RECOVERY_DAYS]
    return pd.Timestamp(activation), pd.Timestamp(peak), pd.Timestamp(trough), end


def segment(value, stock_weight, a, b):
    v = value.loc[a:b]
    lr = np.diff(np.log(v.to_numpy()))
    out = {"return": (v.iloc[-1] / v.iloc[0] - 1) * 100, "mdd": (v / v.cummax() - 1).min() * 100,
           "sharpe": lr.mean() / lr.std(ddof=1) * np.sqrt(252)}
    if stock_weight is not None:
        out["exposure"] = stock_weight.loc[a:b].iloc[1:].mean() * 100
    return out


def segments(value, stock_weight, dates):
    _, peak, trough, end = dates
    return {"down": segment(value, stock_weight, peak, trough),
            "recovery": segment(value, stock_weight, trough, end),
            "full": segment(value, stock_weight, peak, end)}


def run_checkpoint(args):
    root, rec = args
    import torch
    from stable_baselines3 import PPO
    from finrl.experiments import scenario_mix as S
    from finrl.experiments.scenario_mix_common import TICKERS

    torch.set_num_threads(1)
    market = S.MarketData(PRICES, TICKERS)
    run = root / rec["batch"] / f"{rec['group']}_seed{rec['seed']}"
    model = PPO.load(run / "ckpt" / f"step_{rec['timesteps']:06d}.zip", device="cpu")
    keys = {k: rec[k] for k in ("group", "reward", "seed", "timesteps")}
    rows, dailies = [], []
    for name, (a, p, t) in EVENTS.items():
        dates = event_dates(market.dates, a, p, t)
        daily = S.run_continuous(lambda obs: model.predict(obs, deterministic=True)[0],
                                 market, a, dates[3], COMMISSION).set_index("date")
        for seg, met in segments(daily.value, 1 - daily.cash_weight, dates).items():
            rows.append({**keys, "event": name, "segment": seg, **met})
        dailies.append(daily.reset_index().assign(event=name, **keys))
    return rows, pd.concat(dailies)


def main(root, out):
    from finrl.experiments import scenario_mix as S
    from finrl.experiments.scenario_mix_common import TICKERS

    out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(root / "selected_metrics.csv")
    recs = sel[sel.rule == "val_sharpe"][["batch", "group", "reward", "seed", "timesteps"]].to_dict("records")
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(run_checkpoint, [(root, r) for r in recs]))
    by_seed = pd.DataFrame([row for rows, _ in res for row in rows])
    by_seed.to_csv(out / "event_metrics_by_seed.csv", index=False)
    pd.concat([d for _, d in res]).to_csv(out / "event_daily.csv.gz", index=False)
    summary = by_seed.groupby(["event", "segment", "group", "reward"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "event_summary.csv", index=False)

    market = S.MarketData(PRICES, TICKERS)
    bh = []
    for name, (a, p, t) in EVENTS.items():
        dates = event_dates(market.dates, a, p, t)
        value = S.buy_and_hold(market, a, dates[3], COMMISSION, include_cash=False).set_index("date").value
        for seg, met in segments(value, None, dates).items():
            bh.append({"event": name, "segment": seg, "end": dates[3].date(), **met, "exposure": 100.0})
    pd.DataFrame(bh).to_csv(out / "buy_and_hold_events.csv", index=False)
    print(f"{len(recs)} checkpoints x {len(EVENTS)} events -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
