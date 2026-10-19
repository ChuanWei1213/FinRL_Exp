"""Describe the observed seed-8 late-training jumps on validation only."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS


def main():
    torch.set_num_threads(1)
    market = S.MarketData(A.PRICES, TICKERS)
    output = A.OUT / "diagnostics"
    output.mkdir(exist_ok=True)
    rows = []
    for group in A.GROUPS:
        chosen = A.selected_row(group, 8)
        for stage, nominal in [("selected_sharpe", int(chosen["timesteps"])), ("final", 500000)]:
            v = A.read_validation(group, 8)
            expected = v[v.timesteps == nominal].iloc[0]
            checkpoint = A.run_path(group, 8) / "ckpt" / f"step_{nominal:06d}.zip"
            if stage == "selected_sharpe":
                daily = pd.read_csv(A.OUT / "daily" / f"{group}_seed8_validation.csv.gz")
            else:
                daily_file = output / f"{group}_seed8_final_validation.csv.gz"
                meta_file = output / f"{group}_seed8_final_validation.json"
                if daily_file.exists() and meta_file.exists():
                    saved = json.loads(meta_file.read_text())
                    assert saved["checkpoint_sha256"] == A.sha(checkpoint)
                    assert saved["daily_sha256"] == A.sha(daily_file)
                    daily = pd.read_csv(daily_file)
                else:
                    model = PPO.load(checkpoint, device="cpu")
                    daily = A.validation_rollout(model, market, .0025)
                    daily.to_csv(daily_file, index=False, compression={"method":"gzip", "mtime":0})
                    A.write_json(meta_file, {"checkpoint":str(checkpoint.relative_to(A.ROOT)),"checkpoint_sha256":A.sha(checkpoint),"daily_sha256":A.sha(daily_file),"validation_only":True})
            m = A.metrics(daily)
            assert np.isclose(m["return_pct"], (expected.valid_final_value-1)*100, rtol=1e-6, atol=1e-6)
            assert np.isclose(m["sharpe"], expected.valid_sharpe, rtol=1e-6, atol=1e-6)
            weight_cols = [c for c in daily if c.startswith("weight_") and c != "weight_CASH"]
            means = daily[weight_cols].mean().sort_values(ascending=False)
            top = means.index[0].removeprefix("weight_")
            targets = daily[f"target_{top}"].iloc[1:]
            rows.append({"group":group,"seed":8,"stage":stage,"nominal_timesteps":nominal,"largest_mean_asset":top,"largest_asset_mean_weight_pct":float(means.iloc[0]*100),"largest_asset_target_weight_sd_pp":float(targets.std()*100),**m})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "seed8_selected_vs_final.csv", index=False)
    plot(frame)
    print(frame.to_string(index=False))


def plot(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True,"svg.fonttype":"path"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for group in A.GROUPS:
        v = A.read_validation(group, 8)
        chosen = A.selected_row(group, 8)
        axes[0].plot(v.timesteps/1000, v.valid_sharpe, color=A.COLORS[group], linestyle=A.STYLES[group], label=A.LABELS[group], linewidth=2)
        axes[0].scatter(chosen["timesteps"]/1000, chosen["valid_sharpe"], marker="*", s=160, color=A.COLORS[group], edgecolor="white", zorder=4)
    axes[0].set(xlabel="名義訓練步數（千步）", ylabel="Validation Sharpe", xlim=(25, 500), xticks=[25,100,200,300,400,500])
    axes[0].legend(frameon=False, ncol=3)
    labels = []
    for x, row in enumerate(frame.itertuples()):
        is_final = row.stage == "final"
        axes[1].bar(x, row.largest_asset_mean_weight_pct, width=.62, color=A.COLORS[row.group], alpha=1 if is_final else .35, hatch="" if is_final else "//", edgecolor=A.COLORS[row.group])
        axes[1].text(x, row.largest_asset_mean_weight_pct+2, f"{row.largest_mean_asset}\n{row.largest_asset_mean_weight_pct:.1f}%", ha="center", va="bottom", fontsize=10)
        labels.append(f"{A.LABELS[row.group]}\n" + ("最後 500k" if is_final else "Sharpe 選模"))
    axes[1].set(xticks=range(len(frame)),xticklabels=labels,ylabel="最大平均持股：全期平均權重（%）",ylim=(0,110))
    axes[1].grid(axis="x",visible=False)
    fig.suptitle("Seed 8：checkpoint 選擇與單股集中",fontsize=19,y=.99)
    fig.subplots_adjust(top=.91,bottom=.11,hspace=.38)
    fig.text(.02,.015,"2018–2019 validation｜上圖星號＝各組 Sharpe 選中的 checkpoint；下圖為各模型平均權重最高的股票。\n此圖是指定 seed 的行為診斷；不可將其當作全部 seeds 的平均效果。",fontsize=9,color="#555D65")
    figures=A.OUT/"figures"; figures.mkdir(exist_ok=True)
    for extension in ["png","svg"]:
        fig.savefig(figures/f"06_seed8_checkpoint_diagnostic.{extension}",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
