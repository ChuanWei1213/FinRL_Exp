"""Timing vs static exposure: compare each checkpoint with a constant-mix portfolio
(w stocks equal-weight daily rebalanced, 1-w cash, same 0.25% cost ignored) at its own mean w."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from analysis.scenario_mix.reward_shaping_20261105 import evaluate as E
root = Path(sys.argv[1])
prices = Path("../TimeDiff/datasets/dow10/prices_2004_2025.csv")
close = pd.read_csv(prices, parse_dates=["date"]).pivot(index="date", columns="tic", values="close")
ew = close[E.TICKERS].pct_change().mean(axis=1)
fwd = E.market_forward_returns(prices)
thr = E.thresholds(fwd)
sel = pd.read_csv(root / "selected_metrics.csv")
rows = []
for _, r in sel.iterrows():
    d = pd.read_csv(root / "daily" / r.batch / f"{r.group}_seed{r.seed}" / f"step_{r.timesteps:06d}.csv.gz", parse_dates=["date"]).set_index("date")
    w = (1 - d.cash_weight).mean()
    rr = ew.reindex(d.index).fillna(0).iloc[1:]
    static = np.r_[1, np.cumprod(1 + w * rr.to_numpy())]
    st = pd.Series(static, index=d.index)
    lr = np.diff(np.log(static))
    v = d.value; dates = v.index
    ex = []
    for i, t in enumerate(dates[:-21]):
        if t in fwd.index:
            ex.append((fwd[t], (v.iloc[i+21]/v.iloc[i]) - (st.iloc[i+21]/st.iloc[i])))
    ex = pd.DataFrame(ex, columns=["m", "x"])
    rows.append(dict(rule=r.rule, group=r.group, reward=r.reward, seed=r.seed, w=w*100,
                     sharpe=r.sharpe, static_sharpe=lr.mean()/lr.std(ddof=1)*np.sqrt(252),
                     timing_PR10=ex.x[ex.m <= thr["PR10"]].mean()*100,
                     timing_normal=ex.x[ex.m > thr["PR10"]].mean()*100))
t = pd.DataFrame(rows)
t["sharpe_minus_static"] = t.sharpe - t.static_sharpe
t.to_csv(sys.argv[2], index=False)
order = ["base", "lin", "sq", "cube", "exp", "tail"]
pd.set_option("display.width", 200)
for rule in ("val_sharpe", "last"):
    print(f"\n=== {rule} ===")
    print(t[t.rule == rule].groupby(["group", "reward"])[["w", "sharpe", "static_sharpe", "sharpe_minus_static", "timing_PR10", "timing_normal"]].mean().round(3).reindex(order, level=1).to_string())
