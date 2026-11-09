"""Return, max drawdown and mean stock weight of each selected checkpoint in the two stress windows.

Windows match the earlier scenario-mix reports (analysis/scenario_mix/summarize_scenario_mix.py):
COVID 2020-02-01 .. 2020-06-30 and the 2022 bear market 2022-01-01 .. 2022-10-31. Return and
drawdown start from the last close before the window.

    PYTHONPATH=. python crisis_periods.py RESULTS_ROOT OUT_CSV
"""
import sys
from pathlib import Path

import pandas as pd

from analysis.scenario_mix.reward_shaping_20261105 import evaluate as E
from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS

WINDOWS = {"covid": ("2020-02-01", "2020-06-30"), "bear2022": ("2022-01-01", "2022-10-31")}
ORDER = ["base", "lin", "sq", "cube", "exp", "tail"]


def window_stats(value, weight, start, end):
    before = value[value.index < start]
    v = value[(value.index >= start) & (value.index <= end)]
    v = pd.concat([before.iloc[-1:], v])
    w = weight[(weight.index >= start) & (weight.index <= end)]
    return v.iloc[-1] / v.iloc[0] - 1, -(v / v.cummax() - 1).min(), w.mean()


def main(root, out):
    sel = pd.read_csv(root / "selected_metrics.csv")
    rows = []
    for _, r in sel.iterrows():
        d = pd.read_csv(root / "daily" / r.batch / f"{r.group}_seed{r.seed}" / f"step_{r.timesteps:06d}.csv.gz",
                        parse_dates=["date"]).set_index("date")
        row = dict(rule=r.rule, group=r.group, reward=r.reward, seed=r.seed)
        for name, (a, b) in WINDOWS.items():
            ret, mdd, w = window_stats(d.value, 1 - d.cash_weight, a, b)
            row.update({f"{name}_return_pct": ret * 100, f"{name}_mdd_pct": mdd * 100, f"{name}_stock_weight_pct": w * 100})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    cols = [c for c in t.columns if c.endswith("_pct")]
    print(t.groupby(["rule", "group", "reward"])[cols].mean().round(2).reindex(ORDER, level=2).to_string())

    prices = Path("../TimeDiff/datasets/dow10/prices_2004_2025.csv")
    bh = S.buy_and_hold(S.MarketData(prices, TICKERS), *E.TEST, E.COMMISSION, include_cash=True).set_index("date").value
    for name, (a, b) in WINDOWS.items():
        ret, mdd, _ = window_stats(bh, bh * 0 + 10 / 11, a, b)
        print(f"buy&hold 1/11 {name}: return {ret * 100:.2f}%  mdd {mdd * 100:.2f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
