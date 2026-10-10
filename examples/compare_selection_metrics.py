"""Compare scenario-mix runs that differ only in the checkpoint-selection rule.

Each --run is LABEL=TRAIN_ROOT:TEST_ROOT, where TRAIN_ROOT holds the training run dirs
(<group>_seed<k>/ with *_valid.csv and crash_response_all.csv) and TEST_ROOT holds the
summarized test results (all_runs.csv from summarize_scenario_mix.py).

    python examples/compare_selection_metrics.py \
        --run final_value=results/scenario_mix/v1:results/scenario_mix/v1_test2025 \
        --run sharpe=results/scenario_mix/v2_sharpe:results/scenario_mix/v2_sharpe_test2025 \
        --out results/scenario_mix/compare_selection

Outputs (in --out):
    compare_table.csv          seed-mean of every metric per (period, group, label) + diff vs first label
    selected_checkpoints.csv   chosen timestep / validation value / stock weight per run and label
    training_identical.csv     max |diff| of validation final-value curves between labels (same seed)
    crash_response_compare.csv seed-mean crash_response diagnostic per label
    fig_compare.png            seed means/seeds per group and label for key metrics in crash years and the full period
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["A", "B20", "B50", "B70", "B90"]
METRICS = ["return", "max_drawdown", "worst_21d_return", "ann_vol", "sharpe_0rf", "mean_stock_weight"]
FIG_METRICS = {"return": "Return", "max_drawdown": "Max drawdown", "sharpe_0rf": "Sharpe",
               "mean_stock_weight": "Mean stock weight"}
FIG_PERIODS = ["2020", "2022", "2025", "2020-2025"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order


def parse_runs(specs):
    out = {}
    for spec in specs:
        label, roots = spec.split("=", 1)
        train, test = roots.split(":", 1)
        out[label] = (Path(train), Path(test))
    return out


def valid_curves(train_root: Path) -> pd.DataFrame:
    rows = []
    for f in train_root.glob("*/*_seed*_valid.csv"):
        g, seed = re.match(r"(.+)_seed(\d+)_valid", f.stem).groups()
        v = pd.read_csv(f)
        if "valid_score" not in v or v.valid_score.isna().all():
            v["valid_score"] = v.valid_final_value  # older runs: final_value was the only rule
        rows.append(v.assign(group=g, seed=int(seed)))
    return pd.concat(rows, ignore_index=True)


def selected(curves: pd.DataFrame) -> pd.DataFrame:
    # Training keeps the first checkpoint reaching the max (strict >), i.e. idxmax.
    idx = curves.groupby(["group", "seed"]).valid_score.idxmax()
    cols = ["group", "seed", "timesteps", "valid_score", "valid_final_value"]
    cols += [c for c in ("valid_mean_stock_weight",) if c in curves]
    return curves.loc[idx, cols].reset_index(drop=True)


def figure(table: pd.DataFrame, labels, out: Path):
    import matplotlib.pyplot as plt
    periods = [p for p in FIG_PERIODS if p in set(table.period)]
    fig, axes = plt.subplots(len(FIG_METRICS), len(periods), figsize=(3.6 * len(periods), 2.6 * len(FIG_METRICS)),
                             squeeze=False)
    groups = [g for g in GROUPS if g in set(table.group)]
    x = np.arange(len(groups))
    w = 0.8 / len(labels)
    for i, (m, title) in enumerate(FIG_METRICS.items()):
        for j, p in enumerate(periods):
            ax = axes[i, j]
            seg = table[table.period == p]
            for k, lab in enumerate(labels):
                s = seg[seg.label == lab].set_index(["group", "seed"])[m]
                means = [s.loc[g].mean() for g in groups]
                pos = x + (k - (len(labels) - 1) / 2) * w
                for gi, g in enumerate(groups):
                    ax.scatter(np.full(len(s.loc[g]), pos[gi]), s.loc[g], s=10, color=COLORS[k], alpha=0.45,
                               linewidths=0, zorder=2)
                ax.scatter(pos, means, s=46, marker="D", color=COLORS[k], edgecolors="white", linewidths=1.5,
                           label=f"select by {lab}", zorder=3)
            bh = seg[(seg.group == "buy_hold_1_11")][m].dropna()
            if len(bh):
                ax.axhline(bh.iloc[0], color="#555555", ls="--", lw=1, zorder=1, label="buy & hold 1/11")
            ax.set_xticks(x, groups, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(p, fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Checkpoint selection rule: test metrics by group (diamond = seed mean, dots = seeds)",
                 fontsize=11)
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(names),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL=TRAIN_ROOT:TEST_ROOT")
    ap.add_argument("--out", type=Path, required=True)
    cli = ap.parse_args()
    runs = parse_runs(cli.run)
    labels = list(runs)
    cli.out.mkdir(parents=True, exist_ok=True)

    # Test metrics.
    allr = []
    for lab, (_, test) in runs.items():
        allr.append(pd.read_csv(test / "all_runs.csv").assign(label=lab))
    allr = pd.concat(allr, ignore_index=True)
    allr["period"] = allr.period.astype(str)
    mean = allr.groupby(["period", "group", "label"])[METRICS].mean().reset_index()
    base = mean[mean.label == labels[0]].drop(columns="label").set_index(["period", "group"])
    diff = mean.set_index(["period", "group"])[METRICS] - base.reindex(mean.set_index(["period", "group"]).index)
    table = mean.join(diff.reset_index(drop=True).add_prefix(f"diff_vs_{labels[0]}_"))
    table.to_csv(cli.out / "compare_table.csv", index=False)

    # Checkpoint chosen by each rule.
    curves = {lab: valid_curves(train) for lab, (train, _) in runs.items()}
    sel = pd.concat([selected(c).assign(label=lab) for lab, c in curves.items()], ignore_index=True)
    sel.to_csv(cli.out / "selected_checkpoints.csv", index=False)

    # Same seeds -> the training trajectory should be identical; only the pick differs.
    same = []
    for lab in labels[1:]:
        m = curves[labels[0]].merge(curves[lab], on=["group", "seed", "timesteps"], suffixes=("_a", "_b"))
        d = (m.valid_final_value_a - m.valid_final_value_b).abs().groupby([m.group, m.seed]).max()
        same.append(d.rename("max_abs_diff_valid_final_value").reset_index().assign(label=lab))
    if same:
        pd.concat(same).to_csv(cli.out / "training_identical.csv", index=False)

    # Crash-response diagnostic (training-distribution episodes).
    cr = []
    for lab, (train, _) in runs.items():
        f = train / "crash_response_all.csv"
        if f.exists():
            cr.append(pd.read_csv(f).assign(label=lab))
    if cr:
        cr = pd.concat(cr, ignore_index=True)
        cr["scenario"] = cr.scenario.fillna(-1).astype(int)
        crm = cr.groupby(["source", "scenario", "group", "label"])[
            ["pre", "early_crash", "late_crash", "change_late_vs_pre"]].mean().reset_index()
        crm.to_csv(cli.out / "crash_response_compare.csv", index=False)

    figure(allr, labels, cli.out / "fig_compare.png")

    pd.set_option("display.width", 200)
    show = mean[mean.period.isin(FIG_PERIODS)].pivot_table(index=["period", "group"], columns="label",
                                                           values=["return", "max_drawdown", "sharpe_0rf",
                                                                   "mean_stock_weight"])
    print(show.round(3).to_string())
    print("\nselected checkpoints:")
    print(sel.pivot_table(index=["group", "seed"], columns="label", values="timesteps").to_string())
    if same:
        print("\nmax |diff| of validation curves between labels:",
              float(pd.concat(same).max_abs_diff_valid_final_value.max()))
    print(f"-> {cli.out}")


if __name__ == "__main__":
    main()
