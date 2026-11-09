import sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1]); sel = pd.read_csv(root / "selected_metrics.csv")
def seg(v, a, b):
    s = v[(v.index >= a) & (v.index <= b)]
    return (s.iloc[-1] / s.iloc[0] - 1) * 100, -(s / s.cummax() - 1).min() * 100
rows = []
for _, r in sel.iterrows():
    d = pd.read_csv(root / "daily" / r.batch / f"{r.group}_seed{r.seed}" / f"step_{r.timesteps:06d}.csv.gz", parse_dates=["date"]).set_index("date").value
    q1, _ = seg(d, "2019-12-31", "2020-03-31"); _, m20 = seg(d, "2020-01-01", "2020-12-31")
    y22, m22 = seg(d, "2021-12-31", "2022-12-31")
    rows.append(dict(rule=r.rule, group=r.group, reward=r.reward, seed=r.seed, q1_2020=q1, mdd_2020=m20, ret_2022=y22, mdd_2022=m22))
t = pd.DataFrame(rows); t.to_csv(sys.argv[2], index=False)
order = ["base", "lin", "sq", "cube", "exp", "tail"]
print(t.groupby(["rule", "group", "reward"])[["q1_2020", "mdd_2020", "ret_2022", "mdd_2022"]].mean().round(2).reindex(order, level=2).to_string())
