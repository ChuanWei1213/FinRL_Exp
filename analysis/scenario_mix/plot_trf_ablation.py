"""Standalone research figures from the audited TRF comparison CSVs."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

GROUPS = ["A", "B20", "B50", "B70", "B90"]
COLORS = {"legacy": "#2563a8", "fixed": "#d46a1f"}


def style(ax):
    ax.grid(axis="y", color="#e4e7eb", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    cli = ap.parse_args()
    root = cli.report
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    d = pd.read_csv(root / "fixed_checkpoint_deltas.csv", dtype={"period": str})
    d = d[(d.period == "2020-2025") & d.label.isin(["v2_rewardlog", "v3_nocost"])]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharex=True, sharey=True)
    for ax, group in zip(axes, GROUPS):
        for label, color in [("v2_rewardlog", COLORS["legacy"]), ("v3_nocost", COLORS["fixed"])]:
            data = d[(d.group == group) & (d.label == label)]
            for seed, row in data.groupby("seed"):
                row = row.sort_values("actual_timesteps_fixed")
                ax.plot(row.actual_timesteps_fixed/1000, row.delta_return*100, color=color,
                        linewidth=.7, alpha=.35, linestyle=["-", "--", ":"][int(seed)])
            if len(data):
                m = data.groupby("actual_timesteps_fixed").delta_return.mean()
                ax.plot(m.index/1000, m*100, color=color, linewidth=2)
        ax.axhline(0, color="#666", linewidth=.8)
        ax.set(title=group, xlabel="Actual training steps (k)", xlim=(26.624, 532.48))
        ax.set_xticks([26.624, 250, 532.48], labels=["26.6", "250", "532.5"])
        style(ax)
    axes[0].set_ylabel("2020–2025 return change (pp)\nFixed TRF − legacy TRF")
    fig.suptitle("Same historical policy and actions: effect of correcting evaluation costs", fontsize=13, y=1.01)
    fig.legend([Line2D([], [], color=COLORS[k], linewidth=2) for k in COLORS],
               ["v2: trained with costs", "v3: trained without costs"],
               loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False)
    fig.text(.01, -.015, "Thin lines: individual seeds. Thick lines: means of available seeds. Shared y-axis; zero is shown.\nSource: fixed_checkpoint_deltas.csv. These correlated checkpoints are not independent statistical samples.", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .86))
    fig.savefig(root / "fixed_policy_cost_effect.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    r = pd.read_csv(root / "paired_retraining_deltas.csv", dtype={"period": str})
    if r.empty:
        return
    r = r[(r.period == "2020-2025") & (r.rule == "last_step")]
    metrics = [("return", "2020–2025 total return (%)", 100),
               ("max_drawdown", "2020–2025 max drawdown (%)", 100),
               ("sharpe_0rf", "2020–2025 annualized Sharpe (rf = 0)", 1),
               ("mean_stock_weight", "2020–2025 mean stock weight (%)", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (metric, title, scale) in zip(axes, metrics):
        for g, group in enumerate(GROUPS):
            for row in r[r.group == group].itertuples(index=False):
                offset = (row.seed - 1)*.045
                xs = [g-.16+offset, g+.16+offset]
                values = [getattr(row, metric+"_legacy_train")*scale,
                          getattr(row, metric+"_fixed_train")*scale]
                ax.plot(xs, values, color="#9aa2ad", linewidth=.7, zorder=1)
                for j, variant in enumerate(["legacy", "fixed"]):
                    ax.scatter(xs[j], values[j], marker=["o", "s", "^"][row.seed],
                               s=32, color=COLORS[variant], zorder=2)
        ax.set(title=title, xticks=range(5), xticklabels=GROUPS)
        style(ax)
    axes[2].set_ylim(bottom=0)
    axes[3].set_ylim(0, 100)
    fig.suptitle("Paired retraining at the final checkpoint: both policies evaluated with corrected costs", fontsize=12, y=.995)
    handles = [Line2D([], [], color=COLORS[k], marker="o", linewidth=0, label=k+" TRF training") for k in COLORS]
    handles += [Line2D([], [], color="#555", marker=m, linewidth=0, label=f"seed {i}") for i, m in enumerate(["o", "s", "^"])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=5, frameon=False, fontsize=9)
    n_pairs = len(r)
    fig.text(.01, .01, f"Connected points share group and seed; shapes identify seeds. Nominal 500k = actual 532,480 training steps.\nSharpe: daily log-return mean / sample SD × sqrt(252), rf=0; dimensionless.\nSource: paired_retraining_deltas.csv, rule=last_step. {n_pairs}/15 completed seed pairs; no confidence intervals are implied.", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .91), h_pad=2)
    fig.savefig(root / "paired_retraining_final.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
