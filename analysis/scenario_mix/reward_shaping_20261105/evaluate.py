"""Evaluate the reward-shaping study: every checkpoint on 2020-2025, plus market-state windows.

    python -m analysis.scenario_mix.reward_shaping_20261105.evaluate \
        --root results/scenario_mix/reward_shaping_stage1_20261105 \
        --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv

Layout read: <root>/<batch>/<group>_seed<k>/{config.json, valid.csv, ckpt/step_<t>.zip}, where
<batch> is a reward name, optionally with a suffix ("lin__replay") for another dataset batch.
No training. Every rollout is deterministic, continuous and pays the 0.25% commission.

Market-state windows. A window is the 21 trading days after a test date t. Its market return
is the equal-weight, daily-rebalanced 10-stock return over those days (the TimeDiff condition
R). Thresholds are quantiles of the same quantity over training windows whose 21 days lie in
2010-2017 (no test data). For every checkpoint and threshold q in {PR5, PR10}:
    excess_<q>      mean over windows with R <= threshold of (strategy - buy&hold 1/11) return
    excess_normal   same over the windows with R > PR10
    weight_drop_<q> mean over those windows of (stock weight at t - mean stock weight in the window)
Positive excess in down windows with a small negative excess in normal windows is the goal.

Outputs (in --root):
    daily/<batch>/<group>_seed<k>/step_<t>.csv.gz   date, value, cash_weight, turnover
    checkpoint_metrics.csv    one row per (reward, group, seed, nominal step)
    selected_metrics.csv      rows picked by validation Sharpe (earliest on ties) and the last step
    windows.json              thresholds and window counts
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS

HORIZON = 21
TRAIN = ("2010-01-01", "2017-12-31")
TEST = ("2020-01-02", "2025-12-31")
COMMISSION = 0.0025


def market_forward_returns(prices: Path) -> pd.Series:
    """Equal-weight daily-rebalanced return over the 21 trading days after each date."""
    close = pd.read_csv(prices, parse_dates=["date"]).pivot(index="date", columns="tic", values="close")
    daily = close[TICKERS].pct_change().mean(axis=1)
    growth = np.log1p(daily)
    fwd = np.expm1(growth.rolling(HORIZON).sum().shift(-HORIZON))
    return fwd.dropna()


def thresholds(fwd: pd.Series) -> dict:
    a, b = map(pd.Timestamp, TRAIN)
    dates = fwd.index
    # window = days t+1..t+21; all of them inside the training period
    ends = pd.Series(dates, index=dates).shift(-HORIZON)
    ok = (dates >= a) & (ends <= b).fillna(False).to_numpy()
    train = fwd[ok]
    return {"n_train_windows": int(len(train)),
            "PR5": float(train.quantile(0.05)), "PR10": float(train.quantile(0.10))}


def window_stats(daily: pd.DataFrame, bench: pd.Series, fwd: pd.Series, thr: dict) -> dict:
    v = daily.set_index("date").value
    w = 1 - daily.set_index("date").cash_weight
    dates = v.index
    pos = {d: i for i, d in enumerate(dates)}
    rows = []
    for t in fwd.index:
        i = pos.get(t)
        if i is None or i + HORIZON >= len(dates):
            continue
        j = i + HORIZON
        strat = v.iloc[j] / v.iloc[i] - 1
        bh = bench.iloc[j] / bench.iloc[i] - 1
        rows.append((fwd[t], strat - bh, w.iloc[i] - w.iloc[i + 1:j + 1].mean()))
    r = pd.DataFrame(rows, columns=["market", "excess", "weight_drop"])
    out = {"excess_normal_pct": float(r.excess[r.market > thr["PR10"]].mean() * 100)}
    for q in ("PR5", "PR10"):
        down = r[r.market <= thr[q]]
        out[f"excess_{q}_pct"] = float(down.excess.mean() * 100)
        out[f"weight_drop_{q}_pp"] = float(down.weight_drop.mean() * 100)
    return out


def summary(daily: pd.DataFrame) -> dict:
    v = daily.value.to_numpy()
    logr = np.diff(np.log(v))
    years = len(logr) / 252
    roll = pd.Series(v).pct_change(HORIZON).dropna()
    q = roll.quantile(0.05)
    return {"cagr_pct": float(((v[-1] / v[0]) ** (1 / years) - 1) * 100),
            "sharpe": float(logr.mean() / logr.std(ddof=1) * np.sqrt(252)),
            "mdd_pct": float(-(v / np.maximum.accumulate(v) - 1).min() * 100),
            "es95_21d_pct": float(-roll[roll <= q].mean() * 100),
            "stock_weight_pct": float((1 - daily.cash_weight).mean() * 100),
            "annual_turnover": float(daily.turnover.iloc[1:].mean() * 252)}


def eval_run(run: Path, root: Path, prices: Path, thr: dict):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    group, seed = re.match(r"(.+)_seed(\d+)$", run.name).groups()
    batch = run.parent.name
    reward = batch.split("__")[0]
    market = S.MarketData(prices, TICKERS)
    fwd = market_forward_returns(prices)
    fwd = fwd[(fwd.index >= pd.Timestamp(TEST[0])) & (fwd.index <= pd.Timestamp(TEST[1]))]
    bench = S.buy_and_hold(market, *TEST, COMMISSION, include_cash=True).set_index("date").value
    out_dir = root / "daily" / batch / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ck in sorted((run / "ckpt").glob("step_*.zip")):
        step = int(ck.stem.split("_")[1])
        model = PPO.load(ck, device="cpu")
        daily = S.run_continuous(lambda obs: model.predict(obs, deterministic=True)[0],
                                 market, *TEST, COMMISSION)
        daily.to_csv(out_dir / f"step_{step:06d}.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
        rows.append({"reward": reward, "batch": batch, "group": group, "seed": int(seed),
                     "timesteps": step,
                     **summary(daily), **window_stats(daily, bench, fwd, thr)})
    return rows


def select(metrics: pd.DataFrame, root: Path) -> pd.DataFrame:
    picks = []
    for (batch, group, seed), m in metrics.groupby(["batch", "group", "seed"]):
        valid = pd.read_csv(root / batch / f"{group}_seed{seed}" / "valid.csv")
        best = int(valid.loc[valid.valid_sharpe.idxmax(), "timesteps"])   # idxmax = earliest tie
        last = int(m.timesteps.max())
        for rule, step in (("val_sharpe", best), ("last", last)):
            picks.append(m[m.timesteps == step].assign(rule=rule))
    return pd.concat(picks, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=15)
    cli = ap.parse_args(argv)
    fwd = market_forward_returns(cli.prices)
    thr = thresholds(fwd)
    test = fwd[(fwd.index >= pd.Timestamp(TEST[0])) & (fwd.index <= pd.Timestamp(TEST[1]))]
    thr.update({f"test_windows_{q}": int((test <= thr[q]).sum()) for q in ("PR5", "PR10")})
    thr["test_windows_total"] = int(len(test))
    (cli.root / "windows.json").write_text(json.dumps(thr, indent=2) + "\n")

    runs = [run for batch in sorted(cli.root.iterdir())
            if batch.is_dir() and batch.name not in ("daily", "inputs")
            for run in sorted(batch.iterdir()) if (run / "ckpt").is_dir()]
    with ProcessPoolExecutor(cli.workers) as ex:
        futs = [ex.submit(eval_run, run, cli.root, cli.prices, thr) for run in runs]
        rows = [row for f in futs for row in f.result()]
    metrics = pd.DataFrame(rows).sort_values(["reward", "group", "seed", "timesteps"])
    metrics.to_csv(cli.root / "checkpoint_metrics.csv", index=False)
    select(metrics, cli.root).to_csv(cli.root / "selected_metrics.csv", index=False)
    print(f"{len(runs)} runs, {len(metrics)} checkpoints -> {cli.root}")
    print(json.dumps(thr))


if __name__ == "__main__":
    main()
