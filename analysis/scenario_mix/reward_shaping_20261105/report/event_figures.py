"""Market return and stock exposure around the two stress events (reads event_segments.py output).

    PYTHONPATH=. python event_figures.py REPORT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

from analysis.scenario_mix.reward_shaping_20261105.report.event_segments import (
    COMMISSION, EVENTS, PRICES, event_dates)
from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb"})
INK, INK2 = "#0b0b0b", "#52514e"
ORDER = ["base", "lin", "sq", "cube", "exp", "tail"]
COL = dict(zip(ORDER, ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
GROUPS = {"REAL_ONLY": "REAL_ONLY（只用真實）", "REAL_REPLAY_PR30_B50": "真實回放 PR30",
          "PR30_B50": "PR30_B50（合成）", "V3_B50": "V3_B50（舊固定情境）"}
TITLES = {"covid2020": "2020 疫情", "bear2022": "2022 熊市"}


def main(out):
    daily = pd.read_csv(out / "event_daily.csv.gz", parse_dates=["date"])
    daily["exposure"] = (1 - daily.cash_weight) * 100
    market = S.MarketData(PRICES, TICKERS)
    for k, (name, (a, p, t)) in enumerate(EVENTS.items()):
        dates = event_dates(market.dates, a, p, t)
        mkt = S.buy_and_hold(market, a, dates[3], COMMISSION, include_cash=False).set_index("date").value
        mkt = (mkt / mkt.iloc[0] - 1) * 100
        fig = plt.figure(figsize=(13, 9.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.15, 1.15])
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(mkt.index, mkt.values, color=INK, lw=2)
        ax0.set_ylabel("市場累積報酬（%）")
        ax0.set_title(f"{TITLES[name]}｜10 檔等權買入持有（啟用日起算）", loc="left", fontsize=10.5, color=INK)
        axes = [fig.add_subplot(gs[1 + i // 2, i % 2], sharex=ax0) for i in range(4)]
        ev = daily[daily.event == name]
        for ax, g in zip(axes, GROUPS):
            x = ev[ev.group == g]
            for r in ORDER:
                m = x[x.reward == r].groupby("date").exposure.mean()
                ax.plot(m.index, m.values, color=COL[r], lw=1.8, label=r)
            ax.set_title(GROUPS[g], loc="left", fontsize=10, color=INK)
            ax.set_ylabel("股票曝險（%）")
        for ax in [ax0, *axes]:
            for d, lab in ((dates[1], "高點"), (dates[2], "谷底")):
                ax.axvline(d, color=INK2, lw=1, ls="--")
            ax.tick_params(axis="x", labelsize=8.5)
        ax0.text(dates[1], ax0.get_ylim()[1], " 高點", va="top", fontsize=8.5, color=INK2)
        ax0.text(dates[2], ax0.get_ylim()[1], " 谷底", va="top", fontsize=8.5, color=INK2)
        axes[0].legend(frameon=False, fontsize=8.5, ncol=6, loc="lower left")
        fig.text(0.01, 0.005, "曝險為收盤價格漂移後的股票權重，線為 5 seeds 平均（validation Sharpe 選模）；虛線為固定的高點與谷底日。",
                 fontsize=8.5, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(out / f"{4 + k:02d}_event_{name}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
