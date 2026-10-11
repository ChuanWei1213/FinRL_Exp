"""Reconcile TRF paired evaluations, selection effects, and paired retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scenario_mix.trf_ablation import GROUPS, REPO

METRICS = ["return", "max_drawdown", "sharpe_0rf", "mean_stock_weight", "mean_daily_turnover"]
RULES = {"pick_final_value": "valid_final_value", "pick_sharpe": "valid_sharpe", "last_step": "timesteps"}
LABELS = ["v2_rewardlog", "v3_nocost", "paired_legacy", "paired_fixed"]


def table(frame, columns):
    result = ["| " + " | ".join(columns.values()) + " |",
              "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        result.append("| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], float)
                                        else str(row[k]) for k in columns) + " |")
    return "\n".join(result)


def summarize(root, partial=False):
    dest = root / ("progress" if partial else "report")
    dest.mkdir(exist_ok=True)
    valid, test, completed = [], [], []
    for label in LABELS:
        for run in sorted((root / "evaluations" / label).glob("*_seed*")):
            if not (run / "completed.json").exists():
                continue
            v, t = pd.read_csv(run / "valid_paired.csv"), pd.read_csv(run / "test_paired.csv", dtype={"period": str})
            if len(v) != 40 or len(t) != 320:
                raise ValueError(f"Incomplete evaluation: {run}")
            valid.append(v.assign(label=label)); test.append(t.assign(label=label))
            completed.append((label, run.name))
    if not partial and len(completed) != 60:
        raise ValueError(f"Expected 60 complete evaluations, found {len(completed)}")
    if not valid:
        raise ValueError("No complete evaluations yet")
    v, t = pd.concat(valid, ignore_index=True), pd.concat(test, ignore_index=True)
    vk = ["label", "group", "seed", "timesteps"]
    tk = vk + ["period"]
    assert not v.duplicated(vk + ["trf"]).any()
    assert not t.duplicated(tk + ["trf"]).any()
    v.to_csv(dest / "validation.csv", index=False)
    t.to_csv(dest / "evaluations.csv", index=False)
    fixed = t[t.trf == "fixed"].merge(t[t.trf == "legacy"], on=tk, suffixes=("_fixed", "_legacy"), validate="one_to_one")
    for metric in METRICS:
        fixed["delta_" + metric] = fixed[metric + "_fixed"] - fixed[metric + "_legacy"]
    fixed.to_csv(dest / "fixed_checkpoint_deltas.csv", index=False)

    # Select only on validation, never on test. Keep all four selection/evaluation combinations.
    picks, picked_rows = [], []
    for (label, variant, group, seed), values in v.groupby(["label", "trf", "group", "seed"]):
        values = values.sort_values("timesteps")
        for rule, score in RULES.items():
            chosen = values.loc[values[score].idxmax()]
            selection = {"label": label, "selection_trf": variant, "group": group,
                         "seed": seed, "rule": rule, "timesteps": int(chosen.timesteps)}
            picks.append(selection)
            result = t[(t.label == label) & (t.group == group) & (t.seed == seed) & (t.timesteps == chosen.timesteps)]
            picked_rows.extend({**r, "selection_trf": variant, "rule": rule} for r in result.to_dict("records"))
    picks, picked = pd.DataFrame(picks), pd.DataFrame(picked_rows)
    picks.to_csv(dest / "selected_checkpoints.csv", index=False)
    picked.to_csv(dest / "selected_results.csv", index=False)
    pk = ["label", "group", "seed", "rule"]
    changes = picks[picks.selection_trf == "fixed"].merge(picks[picks.selection_trf == "legacy"], on=pk,
                                                         suffixes=("_fixed", "_legacy"), validate="one_to_one")
    changes["changed"] = changes.timesteps_fixed != changes.timesteps_legacy
    changes.to_csv(dest / "selection_changes.csv", index=False)
    keys = ["label", "selection_trf", "trf", "group", "rule", "period"]
    means = picked.groupby(keys)[METRICS].agg(["mean", "std", "count"])
    means.columns = ["_".join(c) for c in means.columns]
    means.reset_index().to_csv(dest / "selected_group_summary.csv", index=False)

    # Same runtime, same seed/data/budget, correct validation and test costs on BOTH sides.
    common = picked[(picked.selection_trf == "fixed") & (picked.trf == "fixed")]
    rk = ["group", "seed", "rule", "period"]
    retrain = common[common.label == "paired_fixed"].merge(common[common.label == "paired_legacy"],
                    on=rk, suffixes=("_fixed_train", "_legacy_train"), validate="one_to_one")
    for metric in METRICS:
        retrain["delta_" + metric] = retrain[metric + "_fixed_train"] - retrain[metric + "_legacy_train"]
    retrain.to_csv(dest / "paired_retraining_deltas.csv", index=False)
    sharpe_columns = rk + ["timesteps_legacy_train", "timesteps_fixed_train",
        "sharpe_0rf_legacy_train", "sharpe_0rf_fixed_train", "delta_sharpe_0rf"]
    retrain[sharpe_columns].to_csv(dest / "sharpe_by_seed.csv", index=False)
    sharpe = retrain.groupby(["rule", "period", "group"]).agg(
        n=("seed", "size"), old_mean=("sharpe_0rf_legacy_train", "mean"),
        old_std=("sharpe_0rf_legacy_train", "std"), new_mean=("sharpe_0rf_fixed_train", "mean"),
        new_std=("sharpe_0rf_fixed_train", "std"), delta_mean=("delta_sharpe_0rf", "mean"),
        delta_std=("delta_sharpe_0rf", "std"), delta_min=("delta_sharpe_0rf", "min"),
        delta_max=("delta_sharpe_0rf", "max"), improved_seeds=("delta_sharpe_0rf", lambda x: int((x > 0).sum())),
    ).reset_index()
    sharpe.to_csv(dest / "sharpe_comparison.csv", index=False)
    spread = common[(common.rule == "last_step") & (common.period == "2020-2025")].groupby(
        ["label", "group"])[METRICS].agg(["mean", "std", "min", "max", "count"])
    spread.columns = ["_".join(c) for c in spread.columns]
    spread.reset_index().to_csv(dest / "last_step_seed_spread.csv", index=False)
    stress = retrain[(retrain.rule == "last_step") & retrain.period.isin(["2020", "2022", "gfc"])]
    stress = stress.groupby(["period", "group"])[["delta_" + m for m in METRICS]].agg(["mean", "std", "count"])
    stress.columns = ["_".join(c) for c in stress.columns]
    stress.reset_index().to_csv(dest / "stress_period_summary.csv", index=False)

    # Ordered, exactly reconciling pipeline contrast: repricing -> reselection -> retraining.
    stages = [
        ("old", "paired_legacy", "legacy", "legacy"),
        ("repriced", "paired_legacy", "legacy", "fixed"),
        ("reselected", "paired_legacy", "fixed", "fixed"),
        ("new", "paired_fixed", "fixed", "fixed"),
    ]
    decomposition = None
    for stage, label, selection, evaluation in stages:
        rows = picked[(picked.label == label) & (picked.selection_trf == selection) & (picked.trf == evaluation)]
        rows = rows[rk + METRICS].rename(columns={m: f"{stage}_{m}" for m in METRICS})
        decomposition = rows if decomposition is None else decomposition.merge(rows, on=rk, validate="one_to_one")
    for metric in METRICS:
        decomposition[f"pricing_delta_{metric}"] = decomposition[f"repriced_{metric}"] - decomposition[f"old_{metric}"]
        decomposition[f"selection_delta_{metric}"] = decomposition[f"reselected_{metric}"] - decomposition[f"repriced_{metric}"]
        decomposition[f"training_delta_{metric}"] = decomposition[f"new_{metric}"] - decomposition[f"reselected_{metric}"]
        decomposition[f"total_delta_{metric}"] = decomposition[f"new_{metric}"] - decomposition[f"old_{metric}"]
        np.testing.assert_allclose(decomposition[f"total_delta_{metric}"],
            decomposition[f"pricing_delta_{metric}"] + decomposition[f"selection_delta_{metric}"]
            + decomposition[f"training_delta_{metric}"], rtol=0, atol=1e-12)
    decomposition.to_csv(dest / "pipeline_decomposition.csv", index=False)

    # Measure runtime reproduction error separately from changing the fee formula.
    reproduction = []
    for label in LABELS[:2]:
        source = REPO / "results/scenario_mix" / label
        old = pd.read_csv(source / "checkpoint_eval.csv", dtype={"period": str})
        joined = t[(t.label == label) & (t.trf == "legacy")].merge(old,
                     on=["group", "seed", "timesteps", "period"], suffixes=("_rerun", "_saved"), validate="one_to_one")
        for metric in METRICS:
            diff = joined[metric + "_rerun"] - joined[metric + "_saved"]
            reproduction.append({"label": label, "metric": metric, "rows": len(diff),
                                 "max_abs_error": diff.abs().max(), "mean_abs_error": diff.abs().mean()})
        old_valid = []
        for run in sorted(source.glob("*_seed*")):
            if not (run / "valid.csv").exists():
                continue
            config = json.loads((run / "config.json").read_text())
            old_valid.append(pd.read_csv(run / "valid.csv").assign(group=config["group"], seed=config["seed"]))
        joined_v = v[(v.label == label) & (v.trf == "legacy")].merge(pd.concat(old_valid),
                     on=["group", "seed", "timesteps"], suffixes=("_rerun", "_saved"), validate="one_to_one")
        for metric in ["valid_final_value", "valid_reward", "valid_sharpe", "valid_mean_stock_weight"]:
            diff = joined_v[metric + "_rerun"] - joined_v[metric + "_saved"]
            reproduction.append({"label": label, "metric": metric, "rows": len(diff),
                                 "max_abs_error": diff.abs().max(), "mean_abs_error": diff.abs().mean()})
    pd.DataFrame(reproduction).to_csv(dest / "historical_reproduction.csv", index=False)

    pairing = []
    for group in GROUPS:
        for seed in range(3):
            left, right = (root / label / f"{group}_seed{seed}" for label in ["paired_legacy", "paired_fixed"])
            if not all((p / "completed.json").exists() for p in (left, right)):
                continue
            a, b = (pd.read_csv(p / "episodes.csv") for p in (left, right))
            cols = ["timesteps", "length", "source", "scenario", "k"]
            same = a[cols].equals(b[cols])
            pairing.append({"group": group, "seed": seed, "episodes_legacy": len(a),
                            "episodes_fixed": len(b), "same_episode_draws": same})
            if not same:
                raise ValueError(f"Episode pairing failed: {group} seed {seed}")
    pd.DataFrame(pairing).to_csv(dest / "episode_pairing.csv", index=False)

    text = ["# TRF 修正前後：回測與成對重訓", "",
            "**進行中的部分結果，尚未完成全部重訓。**" if partial else "**本輪回測及成對重訓已完成。**", "",
            f"已完成 {len(completed)}/60 組模型系列評估；每組 20 個 checkpoint，共 {len(v)//2} 個 checkpoint。", "",
            "## 比較方法", "",
            "- 舊模型：v2 reward logging 與 v3 各 15 run、各 20 checkpoint，重新驗證 2018–2019，測試 2020–2025 與 GFC（2008-01-02～2009-06-30）。",
            "- 固定模型比較：同一動作序列分別執行修正前後的 TRF；逐步斷言 last_action 一致，價格輸入共用同一資料，隔離成本計算效果。",
            "- 重訓比較：A／B20／B50／B70／B90 × seeds 0／1／2，兩個版本各從頭訓練 500,000 名義步數（532,480 實際步數）。兩邊使用同一個本地執行環境。",
            "- 重訓績效對照的兩邊都用正確 TRF 做 validation、選模與 test；最終 checkpoint 另外比較，避免只有選模差異。",
            "- 原始 checkpoint 使用 PyTorch 2.9.1，這次成對重訓使用 2.13.0；不直接把原始舊訓練與本輪新訓練的差異全歸因於 TRF。",
            "- 舊版 step 取自修正 commit 的 parent，包含原數值型別；正式環境程式不被覆寫。可選 QuantStats 終端圖表停用，不参与 reward 或績效計算。",
            "- v3 訓練費率為零，本輪保留既有模型，重做有費率的驗證與測試；不做多餘的零成本重訓。",
            "- 3 個 seed 的平均與個別結果皆保存；不將三個 seed 的差異當成已確立的普遍效果。", "",
            "## 同一 checkpoint 的成本修正效果", "",
            "下表彙整舊模型全部已完成 checkpoint 的 2020–2025 差異。Δ＝正確 TRF－舊 TRF；報酬、回撤與持股單位為百分點。這不是獨立樣本的顯著性檢定。", ""]
    hist = fixed[(fixed.label.isin(LABELS[:2])) & (fixed.period == "2020-2025")]
    stats = hist.groupby("label").agg(n=("timesteps", "size"),
        mean_return=("delta_return", "mean"), min_return=("delta_return", "min"), max_return=("delta_return", "max"),
        mean_mdd=("delta_max_drawdown", "mean"), max_weight=("delta_mean_stock_weight", lambda x:x.abs().max())).reset_index()
    for col in ["mean_return", "min_return", "max_return", "mean_mdd", "max_weight"]:
        stats[col] *= 100
    text += [table(stats, {"label":"模型來源", "n":"checkpoint 數", "mean_return":"平均 Δ 報酬", "min_return":"最小 Δ 報酬", "max_return":"最大 Δ 報酬", "mean_mdd":"平均 Δ 最大回撤", "max_weight":"最大持股差異"}), "",
             "同一 checkpoint 的 Sharpe 變化如下；Sharpe 與其差值均為無單位數值，不是百分比或百分點。", "",
             table(hist.groupby("label").delta_sharpe_0rf.agg(["mean", "min", "max"]).reset_index(),
                   {"label":"模型來源", "mean":"平均 Δ Sharpe", "min":"最小 Δ Sharpe", "max":"最大 Δ Sharpe"}), "",
             "## 驗證選模是否改變", ""]
    changed = changes[changes.label.isin(LABELS[:2])].groupby(["label", "rule"]).changed.agg(["sum", "count"]).reset_index()
    text += [table(changed, {"label":"來源", "rule":"選模規則", "sum":"選到不同 checkpoint", "count":"run 數"}), "",
             "## 相同環境的成對重訓", ""]
    if retrain.empty:
        text += ["尚無完成的成對重訓及回測，不以早期 checkpoint 推論最終結果。", ""]
    else:
        rt = retrain[retrain.period == "2020-2025"].groupby(["rule", "group"]).agg(
            n=("seed", "size"), old_return=("return_legacy_train", "mean"), new_return=("return_fixed_train", "mean"),
            delta_return=("delta_return", "mean"), delta_mdd=("delta_max_drawdown", "mean"),
            old_stock=("mean_stock_weight_legacy_train", "mean"), new_stock=("mean_stock_weight_fixed_train", "mean")).reset_index()
        for col in ["old_return", "new_return", "delta_return", "delta_mdd", "old_stock", "new_stock"]:
            rt[col] *= 100
        text += ["全期為 2020–2025；報酬與持股為 %，差異為百分點，兩邊測試皆採正確成本。", "",
                 table(rt, {"rule":"選模", "group":"組別", "n":"seed 數", "old_return":"舊 TRF 訓練報酬", "new_return":"新 TRF 訓練報酬", "delta_return":"Δ 報酬", "delta_mdd":"Δ 最大回撤", "old_stock":"舊訓練持股", "new_stock":"新訓練持股"}), ""]
        sp = sharpe[sharpe.period == "2020-2025"].copy()
        sp["old"] = sp.apply(lambda r: f"{r.old_mean:.4f} ± {r.old_std:.4f}", axis=1)
        sp["new"] = sp.apply(lambda r: f"{r.new_mean:.4f} ± {r.new_std:.4f}", axis=1)
        text += ["### Sharpe：三種選模規則", "",
                 "沿用 `sharpe_0rf` 定義：日 log return 的平均／樣本標準差 × √252，無風險利率為 0。先各自計算每個種子的 Sharpe，再取平均與樣本標準差；不是把三個策略合成投資組合後的 Sharpe。", "",
                 "下表為 2020–2025 測試 Sharpe；`pick_sharpe` 只在 2018–2019 validation 選模，不是以測試 Sharpe 挑選。± 為三種子標準差，不是信賴區間。", "",
                 table(sp, {"rule":"選模", "group":"組別", "n":"seed 數", "old":"舊訓練 Sharpe", "new":"新訓練 Sharpe", "delta_mean":"平均 Δ Sharpe", "improved_seeds":"改善 seed 數"}), ""]
    text += ["## 來源與可重現性", "",
             "- [資料與執行環境 manifest](../manifest.json)：輸入 SHA-256、Git commit、套件版本、完整預算。",
             "- [同一 checkpoint 差異](fixed_checkpoint_deltas.csv)、[選模變更](selection_changes.csv)、[所選模型績效](selected_results.csv)。",
             "- [成對重訓差異](paired_retraining_deltas.csv)、[逐組平均與標準差](selected_group_summary.csv)、[episode 抽樣一致性](episode_pairing.csv)。",
             "- [終點模型的跨種子範圍](last_step_seed_spread.csv)、[2020／2022／GFC 終點模型差異](stress_period_summary.csv)。",
             "- [Sharpe 三規則／各期間彙整](sharpe_comparison.csv)、[逐種子 Sharpe](sharpe_by_seed.csv)。",
             "- [整體流程差異分解](pipeline_decomposition.csv)：舊訓練／舊選模／舊回測 → 只改回測 → 再改選模 → 再改重訓；三段差異精確加總為總差異。這是指定順序的算術分解，其他切換順序的分項可能不同。",
             "- [舊結果重現誤差](historical_reproduction.csv)：先量化硬體／套件造成的重現差異，再解讀公式效果。",
             "- 每個評估目錄保存 checkpoint 雜湊及逐日壓縮 CSV；來源資料、舊模型與舊報表未覆寫。",
             "- 平均持股為價格變動後的股票權重平均，不是送入環境前的目標 action；最大回撤以負數表示，Δ 為正代表回撤縮小。",
             "- GFC 早於 FinRL 訓練期間，但 TimeDiff 生成器曾看過 2008 年，不稱為整套流程完全未見過的行情。",
             "- 初始價格資料、reward 步數對齊、抽樣比例與選模定義一致；只比較 TRF 程式差異。", ""]
    if (dest / "analysis.md").exists():
        text[4:4] = ["主要結論與圖表見[實驗解讀](analysis.md)。", ""]
    (dest / "README.md").write_text("\n".join(text).replace("参与", "參與"))
    print(dest / "README.md", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    cli = ap.parse_args()
    summarize(cli.root, cli.partial)


if __name__ == "__main__":
    main()
