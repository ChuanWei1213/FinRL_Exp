import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from analysis.scenario_mix.reward_shaping_20261105 import evaluate as E

root, out = Path(sys.argv[1]), Path(sys.argv[2])
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a998f"
ORDER = ["base", "lin", "sq", "cube", "exp", "tail"]
COL = dict(zip(ORDER, ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
GROUPS = {"REAL_ONLY": "REAL_ONLY（只用真實）", "REAL_REPLAY_PR30_B50": "真實回放 PR30",
          "PR30_B50": "PR30_B50（合成）", "V3_B50": "V3_B50（舊固定情境）"}
RULES = {"val_sharpe": "依 validation Sharpe 選模", "last": "固定最後一步（500k）"}
sel = pd.read_csv(root / "selected_metrics.csv"); ck = pd.read_csv(root / "checkpoint_metrics.csv")
tim = pd.read_csv(out / "timing_vs_static_mix.csv")

# static mix Sharpe as a function of stock weight
prices = Path("../TimeDiff/datasets/dow10/prices_2004_2025.csv")
close = pd.read_csv(prices, parse_dates=["date"]).pivot(index="date", columns="tic", values="close")
ew = close[E.TICKERS].pct_change().mean(axis=1)
ew = ew[(ew.index > pd.Timestamp(E.TEST[0])) & (ew.index <= pd.Timestamp(E.TEST[1]))]
ws = np.linspace(0.15, 1.0, 40)
def sh(w):
    lr = np.log1p(w * ew.to_numpy()); return lr.mean() / lr.std(ddof=1) * np.sqrt(252)
static = [sh(w) for w in ws]

# Figure 1: Sharpe vs stock weight
OFF = {"base": (8, -9, "left"), "lin": (-8, 0, "right"), "sq": (8, 0, "left"), "cube": (8, 0, "left"),
       "exp": (-8, 6, "right"), "tail": (8, 0, "left")}
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True); handles = []
for i, rule in enumerate(RULES):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]; d = sel[(sel.rule == rule) & (sel.group == g)]
        ax.plot(ws * 100, static, color=MUTED, lw=2, ls="--", zorder=1)
        ax.text(ws[1] * 100, static[1] + 0.015, "同持股固定組合", color=INK2, fontsize=8.5)
        ax.scatter([93.1], [0.598], marker="*", s=160, color=INK, zorder=4, edgecolor="#fcfcfb", lw=1.5)
        ax.annotate("買進持有", (93.1, 0.598), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        for r in ORDER:
            x = d[d.reward == r]
            ax.scatter(x.stock_weight_pct, x.sharpe, s=22, color=COL[r], alpha=0.35, lw=0, zorder=2)
            mx, my = x.stock_weight_pct.mean(), x.sharpe.mean()
            ax.scatter([mx], [my], s=80, color=COL[r], edgecolor="#fcfcfb", lw=2, zorder=3)
            if i == 0 and j == 0: handles.append(ax.scatter([], [], s=80, color=COL[r], edgecolor="#fcfcfb", lw=2, label=r))
        ax.set_title(f"{GROUPS[g]}｜{RULES[rule]}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(10, 100); ax.set_ylim(0, 0.8)
        if i == 1: ax.set_xlabel("test 平均股票權重（%）")
        if j == 0: ax.set_ylabel("test Sharpe（2020–2025）")
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.025), fontsize=9.5)
fig.text(0.01, 0.005, "小點＝單一 seed；大點＝5 seeds 平均。虛線：以相同權重固定持有 10 檔等權＋現金、每日再平衡、不計手續費的 Sharpe。", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / "01_sharpe_vs_weight.png", dpi=160); plt.close(fig)

# Figure 2: PR10-window excess, total vs timing
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
x = np.arange(len(ORDER)); bw = 0.38
for i, rule in enumerate(RULES):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]
        a = sel[(sel.rule == rule) & (sel.group == g)].groupby("reward").excess_PR10_pct.mean().reindex(ORDER)
        b = tim[(tim.rule == rule) & (tim.group == g)].groupby("reward").timing_PR10.mean().reindex(ORDER)
        ax.bar(x - bw / 2 - 0.01, a, bw, color="#2a78d6", label="相對買進持有（總超額）")
        ax.bar(x + bw / 2 + 0.01, b, bw, color="#eb6834", label="相對同持股固定組合（擇時）")
        for k in range(len(ORDER)):
            ax.text(x[k] - bw / 2, a.iloc[k] + (0.08 if a.iloc[k] >= 0 else -0.3), f"{a.iloc[k]:+.1f}", ha="center", fontsize=7, color=INK2)
            ax.text(x[k] + bw / 2, b.iloc[k] + (0.08 if b.iloc[k] >= 0 else -0.3), f"{b.iloc[k]:+.1f}", ha="center", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x, ORDER); ax.grid(axis="x", visible=False)
        ax.set_title(f"{GROUPS[g]}｜{RULES[rule]}", fontsize=9.5, color=INK, loc="left")
        if j == 0: ax.set_ylabel("PR10 下跌窗平均超額（百分點）")
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.005, "下跌窗：test 期間未來 21 日等權報酬 ≤ 訓練期 PR10（−3.44%）的 223 個起點；超額為該 21 日策略報酬減去比較組合報酬，5 seeds 平均。", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / "02_crash_excess_decomposition.png", dpi=160); plt.close(fig)

# Figure 3: checkpoint curves
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True, sharey="row")
for j, g in enumerate(GROUPS):
    for i, (m, lab) in enumerate([("stock_weight_pct", "test 平均股票權重（%）"), ("sharpe", "test Sharpe")]):
        ax = axes[i, j]
        for r in ORDER:
            c = ck[(ck.group == g) & (ck.reward == r)].groupby("timesteps")[m].mean()
            ax.plot(c.index / 1000, c.values, color=COL[r], lw=2, label=r)
        if m == "sharpe": ax.axhline(0.598, color=INK, lw=1, ls=":"); ax.text(30, 0.603, "買進持有 0.598", fontsize=8, color=INK2)
        if i == 0: ax.set_title(GROUPS[g], fontsize=10, color=INK, loc="left")
        if i == 1: ax.set_xlabel("名義訓練步數（千步）")
        if j == 0: ax.set_ylabel(lab)
axes[0, 0].legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left")
fig.text(0.01, 0.005, "每 25k 名義步一個 checkpoint，線為 5 seeds 平均；test 曲線僅供事後診斷，不用於選模。", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / "03_checkpoint_curves.png", dpi=160); plt.close(fig)
print("ok")
