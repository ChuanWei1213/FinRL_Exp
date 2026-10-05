"""Merge results/scenario_mix/<run>/<group>_seed<k>/ into one report.

    python examples/summarize_scenario_mix.py --root results/scenario_mix/v1

Outputs (in --root):
    all_runs.csv            one row per (group, seed, period) + buy & hold baselines
    by_group.csv            mean / std / n over seeds for every metric
    validation_curves.csv   validation final value vs timesteps for every run
    fig_equity.png          2020-2022 portfolio value: seed mean (line) and range (band)
    fig_drawdown.png        drawdown over time
    fig_stock_weight.png    total stock weight (1 - cash) over time
    fig_covid.png           2020-02-01 .. 2020-06-30 zoom of value and stock weight
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_ORDER = ["A", "B20", "B50", "buy_hold_1_11", "buy_hold_1_10"]
def period_order(periods):
    periods = sorted(set(map(str, periods)))
    return [p for p in periods if "-" not in p] + [p for p in periods if "-" in p]
KEY_METRICS = ["return", "max_drawdown", "worst_21d_return", "ann_vol", "sharpe_0rf",
               "mean_stock_weight", "mean_daily_turnover"]


def load(root: Path):
    summaries, dailies, valids = [], [], []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        s = run / "summary.csv"
        if not s.exists():
            print(f"skip {run.name}: no summary.csv (still running or failed)")
            continue
        summaries.append(pd.read_csv(s))
        for f in run.glob("*_seed*_test_daily.csv"):
            g, seed = re.match(r"(.+)_seed(\d+)_test_daily", f.stem).groups()
            d = pd.read_csv(f, parse_dates=["date"])
            dailies.append(d.assign(group=g, seed=int(seed)))
        for f in run.glob("*_seed*_valid.csv"):
            g, seed = re.match(r"(.+)_seed(\d+)_valid", f.stem).groups()
            valids.append(pd.read_csv(f).assign(group=g, seed=int(seed)))
        for name in ("buy_hold_1_11", "buy_hold_1_10"):
            f = run / f"{name}_test_daily.csv"
            if f.exists() and not any((d.group == name).all() for d in dailies if len(d)):
                dailies.append(pd.read_csv(f, parse_dates=["date"]).assign(group=name, seed=-1))
    if not summaries:
        raise SystemExit("no finished runs found")
    s = pd.concat(summaries, ignore_index=True)
    # baselines are written by every run; keep one copy
    s = s.drop_duplicates(subset=["group", "seed", "period"])
    return s, pd.concat(dailies, ignore_index=True), (pd.concat(valids, ignore_index=True) if valids else None)


def ordered_groups(present):
    present = list(present)
    extra = sorted(g for g in present if g not in GROUP_ORDER)
    base = [g for g in GROUP_ORDER if g in present]
    return [g for g in base if not g.startswith("buy_hold")] + extra + [g for g in base if g.startswith("buy_hold")]


def aggregate(s: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_METRICS if c in s.columns]
    agg = s.groupby(["period", "group"])[cols].agg(["mean", "std", "count"])
    agg = agg.reindex(pd.MultiIndex.from_product(
        [period_order(s.period.unique()),
         ordered_groups(s.group.unique())], names=["period", "group"]))
    return agg


def print_table(agg: pd.DataFrame):
    show = ["return", "max_drawdown", "worst_21d_return", "sharpe_0rf", "mean_stock_weight"]
    for period in agg.index.get_level_values(0).unique():
        print(f"\n=== {period} ===  (mean ± std over seeds)")
        rows = []
        for g in agg.loc[period].index:
            r = {"group": g}
            for m in show:
                if (m, "mean") not in agg.columns:
                    continue
                mu, sd = agg.loc[(period, g), (m, "mean")], agg.loc[(period, g), (m, "std")]
                r[m] = "—" if pd.isna(mu) else (f"{mu:+.3f}" if pd.isna(sd) else f"{mu:+.3f} ± {sd:.3f}")
            rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))


def plots(d: pd.DataFrame, root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = d.copy()
    d["drawdown"] = d.groupby(["group", "seed"]).value.transform(lambda v: v / v.cummax() - 1)
    if "cash_weight" in d:
        d["stock_weight"] = 1 - d["cash_weight"]
    groups = ordered_groups(d.group.unique())

    def band(ax, col, data, title):
        for g in groups:
            x = data[data.group == g]
            if col not in x or x[col].isna().all():
                continue
            st = x.groupby("date")[col].agg(["mean", "min", "max"])
            ls = "--" if g.startswith("buy_hold") else "-"
            ax.plot(st.index, st["mean"], ls, label=g, lw=1.4)
            if x.seed.nunique() > 1:
                ax.fill_between(st.index, st["min"], st["max"], alpha=.15)
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.legend()

    for col, name, title in [("value", "fig_equity.png", "Portfolio value over the test period (start = 1, after build-up cost)"),
                             ("drawdown", "fig_drawdown.png", "Drawdown"),
                             ("stock_weight", "fig_stock_weight.png", "Total stock weight (1 - cash)")]:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        band(ax, col, d, title)
        fig.tight_layout()
        fig.savefig(root / name, dpi=130)
        plt.close(fig)

    covid = d[(d.date >= "2020-02-01") & (d.date <= "2020-06-30")]
    if covid.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    band(axes[0], "value", covid, "COVID window: portfolio value")
    band(axes[1], "stock_weight", covid, "COVID window: total stock weight")
    fig.tight_layout()
    fig.savefig(root / "fig_covid.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    s, d, v = load(root)
    s.to_csv(root / "all_runs.csv", index=False)
    agg = aggregate(s)
    agg.to_csv(root / "by_group.csv")
    if v is not None:
        v.to_csv(root / "validation_curves.csv", index=False)
        col = "valid_score" if "valid_score" in v else "valid_final_value"
        best = v.loc[v.groupby(["group", "seed"])[col].idxmax()]
        show = [c for c in ["group", "seed", "timesteps", "valid_final_value", "valid_score",
                            "valid_stock_weight_std"] if c in best]
        print("\nselected checkpoints (timesteps of best validation):")
        print(best[show].to_string(index=False))
    crash = [pd.read_csv(f) for f in root.glob("*/*_crash_response.csv")]
    if crash:
        c = pd.concat(crash, ignore_index=True)
        c.to_csv(root / "crash_response_all.csv", index=False)
        print("\ncrash response (mean total stock weight; training-period episodes, k=10):")
        print(c.groupby(["group", "source", "scenario"], dropna=False)[
            ["pre", "early_crash", "late_crash", "change_late_vs_pre"]].mean().round(4).to_string())
    print_table(agg)
    plots(d, root)
    print(f"\nwrote all_runs.csv, by_group.csv, figures to {root}")


if __name__ == "__main__":
    main()
