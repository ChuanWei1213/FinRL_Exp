# Scenario-mix 結果分析

此目錄放置已完成實驗的彙整、比較、繪圖與研究 notebook。訓練、模型選擇與重新執行策略回測請使用 [正式實驗入口](../../finrl/experiments/README_scenario_mix.md)；共用紀錄對齊仍由 `finrl.experiments.scenario_mix_reporting` 提供。

## 分析入口

以下命令在專案根目錄執行，後面接原本相同的參數。也可直接執行 `python analysis/scenario_mix/<檔名>.py`。這些是專案內的分析工具，不納入 FinRL 的發行套件。

| 工具 | 正式命令 | 輸入與輸出 |
|---|---|---|
| 彙整各組／seed | `python -m analysis.scenario_mix.summarize_scenario_mix` | `--root` 下的 run 結果；彙整 CSV 與圖仍寫入同一個 `--root` |
| 比較選模規則 | `python -m analysis.scenario_mix.compare_selection_metrics` | 重複指定 `--run LABEL=TRAIN_ROOT:TEST_ROOT`；結果寫入 `--out` |
| 比較所有 checkpoint | `python -m analysis.scenario_mix.compare_checkpoint_runs` | 重複指定 `--run LABEL=ROOT`，每個 ROOT 含訓練紀錄與 `checkpoint_eval.csv`；結果寫入 `--out` |
| Reward 與選模圖 | `python -m analysis.scenario_mix.plot_reward_curves` | `--root` 下的 `episodes.csv`／`valid.csv`；圖與 `picked_steps.csv` 仍寫入同一個 `--root` |

`examples/` 下同名的四支腳本保留為轉接入口，命令列參數與原本的公開分析函式、常數仍可使用。相對輸入／輸出路徑仍以執行時的工作目錄為準。

例如，對既有結果的**工作副本**執行：

```bash
venv/bin/python -m analysis.scenario_mix.summarize_scenario_mix \
  --root /private/tmp/scenario_mix_work/v1

venv/bin/python -m analysis.scenario_mix.compare_selection_metrics \
  --run final_value=/private/tmp/scenario_mix_work/v1:/private/tmp/scenario_mix_work/v1_test2025 \
  --run sharpe=/private/tmp/scenario_mix_work/v2_sharpe:/private/tmp/scenario_mix_work/v2_sharpe_test2025 \
  --out /private/tmp/scenario_mix_work/compare_selection

venv/bin/python -m analysis.scenario_mix.plot_reward_curves \
  --root /private/tmp/scenario_mix_work/v2_rewardlog

venv/bin/python -m analysis.scenario_mix.compare_checkpoint_runs \
  --run train_cost=/private/tmp/scenario_mix_work/v2_rewardlog \
  --run train_nocost=/private/tmp/scenario_mix_work/v3_nocost \
  --out /private/tmp/scenario_mix_work/compare_train_cost
```

這些路徑是範例，需先準備工作副本。工具仍會依原行為覆寫同名衍生 CSV／圖表；搬移本身沒有重建任何歷史報表。

## TRF 修正前後的成對實驗

`run_trf_ablation.py` 在不同 process 執行舊模型重測，以及新舊 TRF 各 15 組的完整重訓。舊版交易 `step` 取自 `8fa237e3` 的 parent，只在實驗 process 內暫時切換，不改寫正式環境或歷史結果。

```bash
MPLCONFIGDIR=/private/tmp/finrl-trf-mpl \
XDG_CACHE_HOME=/private/tmp/finrl-trf-cache \
venv/bin/python -m analysis.scenario_mix.run_trf_ablation \
  --prices ../TimeDiff/datasets/dow10/prices_2004.csv \
  --test-prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv \
  --synthetic ../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz \
  --out results/scenario_mix/trf_ablation_20261011 --workers 8

venv/bin/python -m analysis.scenario_mix.summarize_trf_ablation \
  --root results/scenario_mix/trf_ablation_20261011

venv/bin/python -m analysis.scenario_mix.plot_trf_ablation \
  --report results/scenario_mix/trf_ablation_20261011/report

venv/bin/python -m analysis.scenario_mix.audit_trf_ablation \
  --root results/scenario_mix/trf_ablation_20261011
```

亦可指定 `--stage historical` 或 `--stage training`。已完成的工作以 `completed.json` 辨識並跳過；未完成訓練不會假裝能從缺少 RNG 狀態的 checkpoint 精確接續，需使用新輸出位置從頭重跑。`manifest.json` 記錄資料雜湊、套件版本與預算，拒絕在不同設定下混用同一輸出目錄。

上述訓練命令記錄本輪的輸出位置；提交後的 Git HEAD 已不同，若要重新執行整批訓練，請將 `--out` 改成新目錄，後續彙整命令的 `--root`／`--report` 也指向該目錄。檢查與重新彙整已保存結果則可直接使用本輪目錄。

`trf_ablation.py` 提供單組 `train`／`evaluate` 入口。回測只推論一次動作，再以兩種成本程式執行相同動作，逐步檢查上一個目標 action 一致；新公式結果另以測試確認等同正式 `run_continuous`。可選 QuantStats 終端圖表停用，避免其 SciPy 套件影響無圖形的 worker；訓練 reward 與回測指標仍由原 NumPy／Pandas 程式計算。

輸出依序分為 `paired_legacy`／`paired_fixed` 訓練、`evaluations` 逐 checkpoint 評估與每日 CSV、`report` 最終比較。未完成時可在彙整命令加 `--partial`，產生明確標示為部分結果的 `progress/README.md`；正式報告要求全部 60 組模型系列評估完成，不能用進度報告代替。

`audit_trf_ablation.py` 在完整批次結束後檢查輸入與模型雜湊、ZIP 完整性、模型內的實際步數、episode 步數、成對抽樣序列及報表完整性，成功時寫入 `audit.json`。版本管理保留新訓練 checkpoint、逐日壓縮 CSV、指標彙整與最終報表，讓完整性及 Sharpe 檢查所需的結果一併保存；工作鎖、進度報告與 worker log 留在本地。原始價格與合成資料仍依 manifest 所記錄的路徑及 SHA-256 取得。

Sharpe 比較另輸出 `report/sharpe_comparison.csv`（各組、選模規則與期間的平均、標準差及改善種子數）與 `report/sharpe_by_seed.csv`。沿用原本的日 log-return、年化 252 日、零無風險利率定義；Sharpe 與其差值均無單位。終點模型圖同時呈現報酬、最大回撤、Sharpe 與持股。

## 研究 notebook

主要檔案為 [`scenario_mix_v1_summary.ipynb`](scenario_mix_v1_summary.ipynb)。舊的 [`analysis/scenario_mix_v1_summary.ipynb`](../scenario_mix_v1_summary.ipynb) 是指向它的相對符號連結，因此兩個路徑開啟的是同一份 notebook，沒有另一份需要同步的分析實作。

可從專案根目錄、`analysis/` 或本目錄開啟；原本向上尋找專案根目錄的程式保持不變。`TIMEDIFF_ROOT` 的用法、缺少 TimeDiff 原始資料時的記錄值備援、cell 順序與第一輪研究範圍均保留。

Notebook 產出的 CSV／圖仍寫入 `analysis/scenario_mix_v1_export/`，不隨原始碼一起搬移。TRF 與舊績效的限制仍見 [修正說明](../../examples/README_scenario_mix_corrections.md)；此次期間修正沒有重訓、重新回測或改寫歷史結果。

### 第一輪分析的期間一致性

此 notebook 固定分析 **2020–2022 第一輪結果**。`SRC['FinRL v1']` 已修正為 `results/scenario_mix/v1/`，讓績效表、持股統計、曲線與文字說明使用相同期間。`v1_test2025/` 仍列於來源清單供核對，但不納入第一輪分析。

第 8 節會逐一檢查各 run 的 `summary.csv`，要求標籤為 `2020`、`2021`、`2022`、`2020-2022`。若缺少標籤或混入延長期間，會在彙整前以 `ValueError` 指出來源檔案、預期及實際標籤，避免空表造成不明確的 `KeyError`。不會把 `2020-2025` 的績效改名為 `2020-2022`。

第一輪 validation 紀錄含有新舊兩種格式。第 10 節在合併前，對沒有 `valid_score` 欄位的舊檔使用當時的 `valid_final_value` 作為選模分數；已有的 `valid_score` 保持不變，同分時仍選最早的 checkpoint。

因主來源修正而可能過期的 FinRL 輸出已清除，重新執行即可產生一致的表格與圖；TimeDiff 相關歷史輸出保持原樣。既有結果 CSV、模型與圖檔均未更動。

## 驗證

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/scenario-mix-mpl \
  venv/bin/python -m pytest \
  unit_tests/analysis/test_scenario_mix_tools.py \
  unit_tests/experiments/test_scenario_mix_reporting.py -q
```

測試在暫存目錄產生分析輸入及衍生報表，涵蓋新舊 CLI、相容匯入、seed 彙整、選模同分時的處理、名義／實際步數及 notebook 的根目錄定位。重跑 notebook 應先準備隔離的專案／結果副本，避免其原本的匯出流程覆寫既有檔案。

Notebook smoke test 同時準備第一輪及數值不同的延長測試資料，確認表格與曲線只使用第一輪來源；另涵蓋缺少總期間與混入 2025 結果的錯誤。由同一個 process 內的 IPython 依原順序執行所有 cell，共享狀態並在錯誤時停止；走的是缺少 TimeDiff 原始檔時的記錄值備援路徑。這不等同於真實 Jupyter kernel／前端驗證，也未執行需要 TimeDiff 原始路徑的可選分析。測試不會重寫 notebook 已儲存的輸出。

搬移驗證另以歷史資料副本執行原始與新位置的四支工具，18 個產生的 CSV／PNG 逐位元組一致；`results/scenario_mix/` 原有的 639 個檔案皆通過搬移前後 SHA-256 核對。
