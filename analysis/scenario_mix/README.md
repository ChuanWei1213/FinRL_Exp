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
