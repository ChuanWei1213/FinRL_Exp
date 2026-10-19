"""Build and execute a read-only notebook from completed experiment artifacts.

Use TimeDiff/.venv-spx/bin/python (nbformat/IPython already installed). Cells only
read saved tables/figures and never launch training, selection or test evaluation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import nbformat as nb
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"results/scenario_mix/comprehensive_20261019"
DEST=Path(__file__).with_name("comparison.ipynb")


def main():
    if not DEST.exists():
        subprocess.run([sys.executable,"/Users/chuanwei/.codex/skills/jupyter-notebook/scripts/new_notebook.py","--kind","experiment","--title","TimeDiff 尾部增權：完整 PPO 對照與 v2、v3 比較","--out",str(DEST)],check=True)
    doc=nb.read(DEST,as_version=4)
    cells=[]
    md=lambda text:cells.append(nb.v4.new_markdown_cell(text))
    code=lambda text:cells.append(nb.v4.new_code_cell(text))
    md("""# TimeDiff 尾部增權：完整 PPO 對照與 v2、v3 比較

比較六個本次組別與 v2、v3 各五個歷史組別。所有模型皆以 **validation Sharpe** 選模；test 不參與選擇。

本次缺少的三個對照已補齊：真實回放 PR30、真實回放 PR10、完整歷史分布條件 TimeDiff。每組 seeds 0–9、5 workers、名義 500,000 步（實際 532,480），每 25,000 名義步保存 checkpoint。

這份 notebook 是**唯讀分析入口**，重新執行只會讀取既有結果。訓練、評估與圖表生成程式在同一資料夾，原實驗檔案不會被覆寫。
""")
    code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

def locate_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "finrl/experiments/scenario_mix.py").exists(): return p
        if (p / "FinRL_Exp/finrl/experiments/scenario_mix.py").exists(): return p / "FinRL_Exp"
    raise FileNotFoundError("請從 FinRL_Exp 或其 analysis 子目錄執行")

ROOT = locate_root()
OUT = ROOT / "results/scenario_mix/comprehensive_20261019"
FIG = OUT / "figures"
summary = pd.read_csv(OUT / "selected_summary.csv")
selected = pd.read_csv(OUT / "selected_metrics.csv")
all_steps = pd.read_csv(OUT / "all_checkpoint_metrics.csv")
config = pd.read_csv(OUT / "group_configurations.csv")
paired = pd.read_csv(OUT / "paired_seed_differences.csv")
manifest = json.loads((OUT / "analysis_manifest.json").read_text())
assert len(all_steps) == 5400 and len(selected) == 270 and len(config) == 16
CURRENT = ["REAL_ONLY", "REAL_REPLAY_PR30_B50", "REAL_REPLAY_PR10_B50", "FULL_B50", "PR30_B50", "PR10_B50"]
ORDER = CURRENT + [f"{v}_{g}" for v in ["V2", "V3"] for g in ["A", "B20", "B50", "B70", "B90"]]

def show_figure(name, width=1450):
    display(Image(filename=str(FIG / f"{name}.png"), width=width))

def metric_table(split, common=False):
    frame = pd.read_csv(OUT / "selected_summary_common_seeds0_2.csv") if common else summary
    z = frame[frame.split == split].set_index("group").loc[ORDER]
    result = pd.DataFrame(index=z.index)
    result["seeds"] = z.n_seeds
    for metric, label in [("return_pct", "累積報酬 %"), ("cagr_net_pct", "年化淨報酬 %"), ("sharpe", "Sharpe"), ("sortino", "Sortino"), ("calmar", "Calmar"), ("mdd_pct", "最大回撤 %"), ("es95_daily_pct", "每日 ES95 %"), ("stock_weight_pct", "平均股票曝險 %")]:
        digits = 3 if metric in ["sharpe", "sortino", "calmar"] else 2
        result[label] = [f"{a:.{digits}f} ± {b:.{digits}f}" for a,b in zip(z[f"{metric}_mean"], z[f"{metric}_sd"])]
    return result

def additional_table(split, kind):
    z = summary[summary.split == split].set_index("group").loc[ORDER]
    measures = {
        "risk": [("volatility_pct", "年化波動 %"), ("var95_daily_pct", "每日 VaR95 %"), ("es95_daily_pct", "每日 ES95 %"), ("worst_21d_return_pct", "最差 21 日報酬 %"), ("var95_21d_pct", "21 日 VaR95 %"), ("es95_21d_pct", "21 日 ES95 %")],
        "allocation": [("annual_turnover", "年化換手 倍"), ("fee_factor_drag_pct", "費用因子拖累 %"), ("stock_weight_pct", "股票曝險 %"), ("mean_largest_stock_weight_pct", "平均最大單股 %"), ("max_single_stock_weight_pct", "期間最大單股 %"), ("concentrated_over80_pct", "單股逾 80% 日數 %"), ("action_bound_fraction_pct", "動作邊界比例 %")],
    }
    result = pd.DataFrame(index=z.index)
    for metric, label in measures[kind]:
        result[label] = [f"{a:.2f} ± {b:.2f}" for a,b in zip(z[f"{metric}_mean"], z[f"{metric}_sd"])]
    return result

display(Markdown(f"資料核對：**{len(config)} 組、90 個 runs、1,800 個 checkpoints、5,400 次 split 評估**。"))
''')
    findings=OUT/"findings.json"
    if findings.exists():
        notes=json.loads(findings.read_text())
        md("## 主要結果\n\n"+"\n\n".join(notes["paragraphs"]))
    else:
        md("## 結果閱讀順序\n\n先比較本次六組的 test 指標與配對差異，再看 train／validation 是否與 test 一致。v2、v3 為歷史參照，需連同設定差異與較少的 seeds 解讀。")
    md("""## 實驗設定與公平比較

- Train：2010–2017；validation：2018–2019；test：2020–2025。各 split 獨立以現金與十檔股票各 1/11 建倉，計入建倉成本。
- 本次 B50 各組的替換機率皆為 50%。真實回放組重用對應 PR30／PR10 合成組的 anchor、抽樣配額與重複權重，把生成未來替換成真實未來。所有訓練行情仍落在 train 期間。
- Full 組的條件全部從完整歷史分布抽取；PR30／PR10 組使用 70% 完整池＋30% 尾部池。完整池本身也包含尾部。
- PR 分位數以十檔股票每日等權組合的 21 日累積報酬定義，波動率為同段每日組合 log return 的年化樣本標準差；歷史視窗、回報與波動率成對保留。B50 下額外尾部來源池約占全部 episodes 的 15%，不代表實際尾部行情只有 15%。
- 新 Full 組與原 PR30、PR10 組各有 1,400 條完整池路徑逐值相同（歷史視窗、條件、生成特徵及 OHLC 皆相同），僅另外 600 條的條件來源改變。
- v2、v3 保留既有訓練模型，全部以共同價格、修正後 TRF 與 0.25% 成本重算。v3 訓練時無交易成本；v2／v3 訓練時使用舊 TRF。它們也使用不同生成器／固定情境與執行環境，不能將差異全部歸因於抽樣。
- 舊固定情境資料共 6,039 條，實際條件為（21 日報酬、年化波動）＝（−10%、50%）、（−23%、100%）、（−36%、140%），各 2,013 條。本次每批 2,000 條，使用另行重訓的同一 TimeDiff checkpoint；生成器訓練期為 2005–2017，條件抽樣期為 2010–2017。
- v2 的完整訓練歷程來自 `v2_rewardlog`；15 個原 `v2_sharpe` 已選模型另以權重逐項核對，避免重複計入同一模型。
""")
    code('''display(config[["group", "n_seeds", "training_trf", "train_commission", "evaluation_commission", "replacement_probability", "replacement_source", "condition_policy", "torch_version"]])
quality = pd.read_csv(OUT / "generation_quality_comparison.csv")
display(quality[["dataset", "paths", "unique_histories", "unique_close_paths", "return_mae_pp", "vol_mae_pp"]].rename(columns={"dataset":"資料批次", "paths":"路徑數", "unique_histories":"不同歷史視窗", "unique_close_paths":"不同收盤路徑", "return_mae_pp":"報酬 MAE 百分點", "vol_mae_pp":"波動 MAE 百分點"}).round(3))
display(quality[["dataset", "realized_pr10_pct", "realized_pr30_pct", "mean_stock_correlation", "paired_real_stock_correlation", "high_correction_pct", "low_correction_pct"]].rename(columns={"dataset":"資料批次", "realized_pr10_pct":"實際 PR10 占比 %", "realized_pr30_pct":"實際 PR30 占比 %", "mean_stock_correlation":"生成股票相關性", "paired_real_stock_correlation":"配對真實相關性", "high_correction_pct":"high 修正 %", "low_correction_pct":"low 修正 %"}).round(3))
show_figure("10_generation_tail_comparison")
''')
    md("""## 各組 train、validation、test 指標

表中為 **seed 平均 ± 樣本標準差**；標準差不是信賴區間。最大回撤採正的損失幅度；VaR95 與 ES95 損失記正、收益記負，越低越好。完整 CSV 另含 Sortino、波動率、換手、費用因子拖累與集中度等指標。

Train 指標是選中模型在真實訓練期間的連續回測，與訓練過程中的隨機 episode reward 不同。期間長度不同，跨 split 比較優先使用年化報酬、Sharpe 與風險指標。
""")
    code('''for split, title in [("train", "Train 2010–2017"), ("val", "Validation 2018–2019"), ("test", "Test 2020–2025")]:
    display(Markdown(f"### {title}"))
    display(metric_table(split))
    display(Markdown("**尾部風險與波動**"))
    display(additional_table(split, "risk"))
    display(Markdown("**交易成本與持股集中度**"))
    display(additional_table(split, "allocation"))
show_figure("03_selected_metrics_test")
''')
    md("""## 訓練步數曲線與選模

每個 checkpoint 都在相同 split 連續評估。粗線為平均，色帶為 seeds 的 25–75% 範圍。**test 曲線僅用來診斷，沒有用 test 最高點挑模型或回頭調整訓練。**

每個 run 只選一個 validation Sharpe 最大的 checkpoint，同分取較早步數；該模型用於 train、validation、test 三個主要表格。最後 500k 的結果另列為敏感度分析。
""")
    code('''show_figure("01_current_checkpoint_curves")
show_figure("02_common_seed_version_curves")
choices = pd.read_csv(OUT / "selected_checkpoints.csv")
display(choices.groupby("group", sort=False).agg(seeds=("seed", "size"), selected_step_median=("nominal_timesteps", "median"), selected_step_min=("nominal_timesteps", "min"), selected_step_max=("nominal_timesteps", "max"), changes_from_original_log=("selection_changed_from_original_log", "sum")))
''')
    md("## Test 配對差異\n\n配對單位是相同 seed，區間僅涵蓋訓練 seed 變異。不同市場時期、生成器訓練種子、條件抽樣種子皆未納入；多重比較未校正，作探索性解讀。")
    code('''show_figure("05_test_paired_differences")
display(paired[(paired.split == "test") & paired.metric.isin(["cagr_net_pct", "sharpe", "mdd_pct", "es95_daily_pct"])].round(4))
show_figure("11_selected_horizon21_risk")
show_figure("12_current_horizon21_curves")
display(summary[summary.split == "test"][["group", "n_seeds", "worst_21d_return_pct_mean", "worst_21d_return_pct_sd", "var95_21d_pct_mean", "es95_21d_pct_mean", "es95_21d_pct_sd"]].round(3))
''')
    md("## 淨值、回撤與跨期間表現\n\n買入持有參照包含現金各 1/11 的版本；另保存十檔股票等權、無初始現金版本。逐日中位數曲線不等於可直接交易的組合。")
    code('''show_figure("04_selected_paths")
show_figure("06_generalization")
show_figure("08_test_annual_heatmap")
display(pd.read_csv(OUT / "benchmark_metrics.csv")[["group", "split", "return_pct", "cagr_net_pct", "sharpe", "mdd_pct", "es95_daily_pct"]].round(3))
subperiod = pd.read_csv(OUT / "selected_subperiod_metrics.csv")
display(Markdown("**Test 前後半段：各 seed 指標的平均**"))
halves = subperiod[(subperiod.split == "test") & subperiod.period.isin(["2020-2022", "2023-2025"])]
display(halves.pivot_table(index="group", columns="period", values=["cagr_net_pct", "sharpe", "mdd_pct"]).reindex(ORDER).round(3))
display(Markdown("**2020 Q1 壓力期間：累積報酬與風險（沒有將季度報酬年化）**"))
display(subperiod[(subperiod.split == "test") & (subperiod.period == "2020Q1")].groupby("group")[["return_pct", "mdd_pct", "es95_21d_pct", "stock_weight_pct"]].mean().reindex(ORDER).round(3))
''')
    md("""## 持股比例與集中風險

以下先展示本次六組的完整持股時間圖。每組三欄分別為 train、validation、test；上排是跨 seeds 平均實際收盤權重，下排呈現總曝險與最大單股權重的 seeds 中位數及 IQR。完整 v2、v3 圖也在下方索引。

最大單股權重先逐 seed／逐日計算，再彙整，避免不同 seeds 集中在不同股票時，被平均權重掩蓋。
""")
    code('''show_figure("09_exposure_concentration")
for group in CURRENT:
    show_figure(f"by_group/holdings_{group}")
display(Markdown("**Test 單股權重曾超過 60% 的 seeds**：僅作持股診斷，所有 seeds 均保留在績效統計內。"))
asset = pd.read_csv(OUT / "selected_asset_weights.csv")
asset = asset[(asset.split == "test") & (asset.asset != "CASH")].sort_values("mean_weight_pct", ascending=False).drop_duplicates(["group", "seed"])
concentrated = selected[(selected.split == "test") & (selected.max_single_stock_weight_pct > 60)].merge(asset[["group", "seed", "asset", "mean_weight_pct"]], on=["group", "seed"], validate="one_to_one")
display(concentrated[["group", "seed", "nominal_timesteps", "asset", "mean_weight_pct", "max_single_stock_weight_pct", "cagr_net_pct", "mdd_pct"]].round(3))
''')
    md("## 真實訓練 episodes 與最後模型敏感度\n\n訓練 episode reward 受抽中的行情、episode 長度和成本設定影響，因此同時提供每交易日正規化版本。v2、v3 的 scenario 是固定壓力情境編號，並非本次的 full／tail 來源池，因此其尾部池比例標示為不適用。")
    code('''show_figure("07_training_episode_rewards")
sampling = pd.read_csv(OUT / "episode_sampling.csv")
display(sampling.groupby("group", sort=False)[["replacement_pct", "synthetic_pct", "real_replay_pct", "tail_pool_all_episodes_pct", "tail_pool_replacement_pct", "mean_length"]].mean().round(3).fillna("不適用"))
final = pd.read_csv(OUT / "fixed_final_summary.csv")
z = summary[summary.split == "test"][["group", "sharpe_mean", "cagr_net_pct_mean", "mdd_pct_mean"]].merge(final[final.split == "test"][["group", "sharpe_mean", "cagr_net_pct_mean", "mdd_pct_mean"]], on="group", suffixes=("_selected", "_500k"))
display(z.round(3))
display(Markdown("### 共同 seeds 0–2 的 test 表格"))
display(metric_table("test", common=True))
''')
    md("## 全部圖表與資料索引\n\n每組皆有績效、風險與成本、持股集中度三套步數曲線，以及持股比例時間圖；每張圖均提供 PNG 與可縮放 SVG。")
    code('''from urllib.parse import quote
rows = []
for group in ORDER:
    links = []
    for prefix, label in [("curves", "績效步數曲線"), ("curves_risk", "風險／成本步數曲線"), ("curves_allocation", "集中度／21日風險曲線"), ("holdings", "持股時間圖")]:
        path = FIG / "by_group" / f"{prefix}_{group}.png"
        assert path.exists()
        links.append(f"[{label}]({path})")
    rows.append(f"| {group} | " + " · ".join(links) + " |")
display(Markdown("| 組別 | 圖表 |\\n|---|---|\\n" + "\\n".join(rows)))
for name in ["selected_metrics.csv", "selected_summary.csv", "all_checkpoint_metrics.csv", "selected_checkpoints.csv", "selected_summary_common_seeds0_2.csv", "paired_seed_differences.csv", "selected_subperiod_metrics.csv", "selected_asset_weights.csv", "checkpoint_selection_sensitivity.csv", "generation_quality_comparison.csv"]:
    display(Markdown(f"- [{name}]({OUT / name})"))
''')
    md("""## 方法與限制

- Sharpe：日 log return 平均／樣本標準差 × √252，無風險利率設為 0。Sortino 的 downside deviation 以全部交易日為分母。
- 年化淨報酬包含建倉成本；為與既有紀錄一致，Calmar 分子沿用平均日 log return 年化值，建倉成本位於初始淨值，不算成單日報酬。
- 最大回撤為每日淨值相對當時歷史高點的最大跌幅；ES95 為最低 5% 簡單日報酬的平均損失。
- 21 日風險採各 split 內重疊的 21 交易日淨值報酬；最差報酬保留報酬原始正負號，VaR／ES 損失記正、收益記負。重疊視窗不當成獨立樣本。
- 年度與 test 子期間延續原持股，沒有逐年重設；只在整個 split 開始時收建倉成本。
- Validation 同時用於 TimeDiff 與 PPO 的選模，具有選擇偏差。v2、v3 的 test 曾在先前實驗被檢視，本比較不宣稱是全新盲測。
- 本次只代表單一股票組合、固定資料期間與單一生成資料批次；重疊視窗、重複條件與多條生成路徑不是獨立市場事件。
- 生成器原始價格檔與 PPO 延伸價格檔存在微小調整價修訂，重疊價格最大相對差約 1.282 ppm；所有 PPO 比較均使用同一延伸價格檔。
- v2、v3 的訓練環境、TRF 與生成器條件不同，統一重算評估可以消除評分公式差異，但無法消除訓練差異。

Notebook 以單一程序 IPython 從頭執行並保存輸出，符合此專案的無 socket 驗證方式；沒有啟動 Jupyter kernel。
""")
    code('''audit = json.loads((OUT / "analysis_manifest.json").read_text())
display(pd.DataFrame({"item": ["groups", "runs", "checkpoints", "split_evaluations", "selected_split_rows"], "count": [len(audit["groups"]), audit["runs"], audit["checkpoints"], audit["split_evaluations"], audit["selected_split_rows"]]}))
display(pd.read_csv(OUT / "v2_saved_model_equivalence.csv")[["group", "seed", "original_sharpe_selected_step", "policy_equal"]])
assert pd.read_csv(OUT / "v2_saved_model_equivalence.csv").policy_equal.all()
''')
    doc.cells=cells
    doc.metadata["kernelspec"]={"display_name":"Python 3", "language":"python", "name":"python3"}
    doc.metadata["language_info"]={"name":"python","version":sys.version.split()[0]}
    shell=InteractiveShell.instance();shell.user_ns["__name__"]="__main__"
    import os
    os.chdir(ROOT)
    count=0
    for cell in doc.cells:
        if cell.cell_type!="code":continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}")
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nb.v4.new_output("stream",name="stdout",text=captured.stdout))
        if captured.stderr:cell.outputs.append(nb.v4.new_output("stream",name="stderr",text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nb.v4.new_output("display_data",data=output.data,metadata=output.metadata))
    nb.validate(doc);nb.write(doc,DEST)
    audit={"executed_at":datetime.now(timezone.utc).isoformat(),"code_cells":count,"execution":"sequential in-process IPython; no kernel/socket","errors":0,
           "embedded_images":sum("image/png" in o.get("data",{}) for c in doc.cells for o in c.get("outputs",[])),"file":str(DEST)}
    (OUT/"notebook_execution.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
