"""Summarise stage-1 reward shaping: per group x reward, mean over seeds, paired diff vs base."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(sys.argv[1])
order = ["base", "lin", "sq", "cube", "exp", "tail"]
cols = ["sharpe", "cagr_pct", "mdd_pct", "es95_21d_pct", "excess_PR5_pct", "excess_PR10_pct",
        "excess_normal_pct", "weight_drop_PR10_pp", "stock_weight_pct", "annual_turnover"]
sel = pd.read_csv(root / "selected_metrics.csv")
ck = pd.read_csv(root / "checkpoint_metrics.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

def table(df, title):
    print(f"\n=== {title} ===")
    for g, d in df.groupby("group"):
        m = d.groupby("reward")[cols].mean().reindex(order)
        sd = d.groupby("reward")[["sharpe", "cagr_pct", "mdd_pct"]].std().reindex(order)
        m.insert(1, "sharpe_sd", sd.sharpe)
        # paired (same seed) difference vs base
        piv = d.pivot_table(index="seed", columns="reward", values="sharpe")
        diff = piv.sub(piv["base"], axis=0)
        m["dSharpe_vs_base"] = diff.mean()
        m["n_seed_better"] = (diff > 0).sum()
        piv5 = d.pivot_table(index="seed", columns="reward", values="excess_PR10_pct")
        m["dExPR10_vs_base"] = piv5.sub(piv5["base"], axis=0).mean()
        print(f"\n-- {g} --"); print(m.round(3).to_string())

for rule in ("val_sharpe", "last"):
    table(sel[sel.rule == rule], f"selection = {rule}")
table(ck, "all 20 checkpoints averaged")
# selected steps
print("\nselected step (val_sharpe), mean over seeds:")
print(sel[sel.rule == "val_sharpe"].groupby(["group", "reward"]).timesteps.agg(list).unstack(0).reindex(order))
