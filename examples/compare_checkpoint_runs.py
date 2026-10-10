"""Compare log_training_rewards.py roots (e.g. training with vs without commission) at every step.

Needs <root>/<group>_seed<k>/{episodes,valid}.csv and <root>/checkpoint_eval.csv
(from evaluate_checkpoints.py).

    python examples/compare_checkpoint_runs.py \
        --run train_cost=results/scenario_mix/v2_rewardlog \
        --run train_nocost=results/scenario_mix/v3_nocost \
        --out results/scenario_mix/compare_train_cost

Outputs (in --out):
    fig_steps.png      per group: training reward, validation reward, test metrics vs steps
                       (thick = seed mean, thin = seeds)
    picked_table.csv   test metrics at the checkpoint picked by each rule and at the last step
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["A", "B20", "B50", "B70", "B90"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
ROWS = [("train_real", None, "Train reward, real ep.\n(stochastic, own costs)"),
        ("valid_reward", None, "Valid reward 2018-19\n100·log(final value)"),
        ("return", "2020-2025", "Test return\n2020-2025"),
        ("max_drawdown", "2020", "Test max drawdown\n2020 (COVID)"),
        ("max_drawdown", "gfc", "Max drawdown\n2008-09 (unseen)"),
        ("mean_stock_weight", "2020-2025", "Test mean\nstock weight"),
        ("mean_daily_turnover", "2020-2025", "Test mean\ndaily turnover")]
TABLE_PERIODS = ["2020", "2022", "2025", "2020-2025", "gfc"]
TABLE_METRICS = ["return", "max_drawdown", "sharpe_0rf", "mean_stock_weight", "mean_daily_turnover"]


def load(root: Path) -> pd.DataFrame:
    """One row per (group, seed, timesteps) with training/validation columns."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "valid.csv").exists()):
        g, seed = re.match(r"(.+)_seed(\d+)$", d.name).groups()
        v = pd.read_csv(d / "valid.csv")
        ep = pd.read_csv(d / "episodes.csv")
        f = int(v.timesteps.iloc[0])
        ep["bucket"] = (np.ceil(ep.timesteps / f) * f).astype(int)
        v["train_real"] = v.timesteps.map(ep[ep.source == "real"].groupby("bucket").reward.mean())
        out.append(v.assign(group=g, seed=int(seed)))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL=ROOT")
    ap.add_argument("--out", type=Path, required=True)
    cli = ap.parse_args()
    runs = dict(s.split("=", 1) for s in cli.run)
    cli.out.mkdir(parents=True, exist_ok=True)
    tr = {lab: load(Path(r)) for lab, r in runs.items()}
    ev = {lab: pd.read_csv(Path(r) / "checkpoint_eval.csv").astype({"period": str}) for lab, r in runs.items()}

    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if any(g in set(t.group) for t in tr.values())]
    fig, axes = plt.subplots(len(ROWS), len(groups), figsize=(3.3 * len(groups), 1.95 * len(ROWS)),
                             sharex=True, sharey="row", squeeze=False)
    for j, g in enumerate(groups):
        for k, lab in enumerate(runs):
            c = COLORS[k]
            for i, (col, period, _) in enumerate(ROWS):
                ax = axes[i, j]
                if period is None:
                    d = tr[lab][tr[lab].group == g]
                else:
                    e = ev[lab]
                    d = e[(e.group == g) & (e.period == period)]
                if d.empty or d[col].isna().all():
                    continue
                for _, s in d.groupby("seed"):
                    ax.plot(s.timesteps / 1000, s[col], color=c, lw=0.7, alpha=0.35)
                m = d.groupby("timesteps")[col].mean()
                ax.plot(m.index / 1000, m.values, color=c, lw=2, label=lab)
        for i, (_, _, ylab) in enumerate(ROWS):
            ax = axes[i, j]
            ax.grid(color="#e5e5e5", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(g, fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            if i == len(ROWS) - 1:
                ax.set_xlabel("training steps (k)", fontsize=9)
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.suptitle("Training with vs without commission: metrics of every checkpoint "
                 "(thick = seed mean, thin = seeds; test/valid always pay 0.25%)", fontsize=11)
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(names),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(cli.out / "fig_steps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Test metrics at the picked checkpoint (first max, as in training) and at the last step.
    rows = []
    for lab in runs:
        t, e = tr[lab], ev[lab]
        picks = {"pick_final_value": t.loc[t.groupby(["group", "seed"]).valid_final_value.idxmax()],
                 "pick_sharpe": t.loc[t.groupby(["group", "seed"]).valid_sharpe.idxmax()],
                 "last_step": t.loc[t.groupby(["group", "seed"]).timesteps.idxmax()]}
        for rule, p in picks.items():
            sel = e.merge(p[["group", "seed", "timesteps"]], on=["group", "seed", "timesteps"])
            sel = sel[sel.period.isin(TABLE_PERIODS)]
            rows.append(sel.assign(label=lab, rule=rule))
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(cli.out / "picked_table.csv", index=False)
    pd.set_option("display.width", 220)
    for period in ["2020-2025", "2022", "gfc"]:
        print(f"\n== {period} (seed mean) ==")
        print(tab[tab.period == period].pivot_table(index="group", columns=["rule", "label"],
                                                    values=["return", "max_drawdown", "mean_daily_turnover"])
              .round(3).to_string())
    print(f"-> {cli.out}")


if __name__ == "__main__":
    main()
