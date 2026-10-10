"""Training reward, validation reward and the selected checkpoints vs training steps.

Reads the output of log_training_rewards.py (<root>/<group>_seed<k>/{episodes,valid}.csv).

    python examples/plot_reward_curves.py --root results/scenario_mix/v2_rewardlog

Rows: training episode reward on real / synthetic episodes (mean per eval interval),
validation reward 100*log(final value), validation Sharpe, validation stock weight.
Markers show the checkpoint each selection rule picks (first max, as in training).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["A", "B20", "B50", "B70", "B90"]
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order
PICKS = {"valid_final_value": ("o", "pick by final value"), "valid_sharpe": ("D", "pick by Sharpe")}
ROWS = [("train_real", "Train reward\nreal episodes"), ("train_syn", "Train reward\nsynthetic episodes"),
        ("valid_reward", "Valid reward\n100·log(final value)"), ("valid_sharpe", "Valid Sharpe"),
        ("valid_mean_stock_weight", "Valid mean\nstock weight")]


def load(root: Path):
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "valid.csv").exists()):
        g, seed = re.match(r"(.+)_seed(\d+)$", d.name).groups()
        v = pd.read_csv(d / "valid.csv")
        ep = pd.read_csv(d / "episodes.csv")
        eval_freq = int(v.timesteps.iloc[0])
        ep["bucket"] = (np.ceil(ep.timesteps / eval_freq) * eval_freq).astype(int)
        for src, col in (("real", "train_real"), ("synthetic", "train_syn")):
            v[col] = v.timesteps.map(ep[ep.source == src].groupby("bucket").reward.mean())
        runs[(g, int(seed))] = v
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    cli = ap.parse_args()
    runs = load(cli.root)
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if any(k[0] == g for k in runs)]
    fig, axes = plt.subplots(len(ROWS), len(groups), figsize=(3.3 * len(groups), 2.1 * len(ROWS)),
                             sharex=True, sharey="row", squeeze=False)
    picks = []
    for j, g in enumerate(groups):
        for (gg, seed), v in sorted(runs.items()):
            if gg != g:
                continue
            c = SEED_COLORS[seed % len(SEED_COLORS)]
            x = v.timesteps / 1000
            sel = {m: v[m].idxmax() for m in PICKS}
            picks.append({"group": g, "seed": seed, **{f"{m}_step": int(v.timesteps[i]) for m, i in sel.items()}})
            for i, (col, _) in enumerate(ROWS):
                ax = axes[i, j]
                if v[col].isna().all():
                    continue
                ax.plot(x, v[col], color=c, lw=1.6, label=f"seed {seed}")
                if col.startswith("valid"):
                    for m, idx in sel.items():
                        mk, lab = PICKS[m]
                        ax.scatter(x[idx], v[col][idx], marker=mk, s=55, facecolor=c if mk == "D" else "white",
                                   edgecolor=c, linewidths=1.6, zorder=4, label=lab)
        for i, (col, ylab) in enumerate(ROWS):
            ax = axes[i, j]
            ax.grid(color="#e5e5e5", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8)
            if col == "train_syn" and g == "A":
                ax.text(0.5, 0.5, "no synthetic episodes", transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#777777")
            if i == 0:
                ax.set_title(g, fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == len(ROWS) - 1:
                ax.set_xlabel("training steps (k)", fontsize=9)
    # Legend: seed colors + the two pick markers (in neutral ink).
    from matplotlib.lines import Line2D
    seeds = sorted({s for _, s in runs})
    handles = [Line2D([], [], color=SEED_COLORS[s % len(SEED_COLORS)], lw=1.6, label=f"seed {s}") for s in seeds]
    handles += [Line2D([], [], ls="", marker="o", mfc="white", mec="#333333", mew=1.6, ms=7,
                       label="checkpoint picked by final value"),
                Line2D([], [], ls="", marker="D", mfc="#333333", mec="#333333", ms=6,
                       label="checkpoint picked by Sharpe")]
    fig.suptitle("Reward and checkpoint selection vs training steps (train reward = mean episode reward per "
                 "25k steps, 100·log return)", fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(handles),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(cli.root / "fig_reward_vs_steps.png", dpi=140, bbox_inches="tight")
    pd.DataFrame(picks).to_csv(cli.root / "picked_steps.csv", index=False)
    print(f"-> {cli.root / 'fig_reward_vs_steps.png'}")


if __name__ == "__main__":
    main()
