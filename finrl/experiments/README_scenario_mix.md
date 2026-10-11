# Scenario-mix 實驗入口

Scenario-mix 的正式訓練與回測由 `finrl.experiments` 提供。以下指令均在專案根目錄執行；相對資料與輸出路徑仍以目前工作目錄為準。

## 模組分工

| 模組 | 責任 |
|---|---|
| `scenario_mix.py` | 資料載入、混合情境環境、連續回測、基準與績效指標 |
| `scenario_mix_common.py` | 固定的股票順序、PPO 建立方式、validation 選模分數 |
| `train_scenario_mix.py` | 依組別／seed 訓練、保存最佳 checkpoint、回測與診斷 |
| `train_scenario_mix_logged.py` | 單一組別／seed 訓練、episode reward 紀錄、保存所有 checkpoint |
| `evaluate_scenario_mix.py` | 對已選定模型重新測試，不重訓或重新選模 |
| `evaluate_scenario_mix_checkpoints.py` | 平行回測所有已保存 checkpoint |
| `scenario_mix_reporting.py` | 既有紀錄的實際步數對齊與 reward 分桶 |

四個正式入口都提供 `build_parser()`、`run(cli)` 與 `main(argv=None)`。CLI 僅負責解析參數後呼叫流程；共用邏輯不再從 `examples/` 匯入。一般訓練入口另保留 `parse(argv=None)`，供舊程式相容使用。

`examples/` 的四個舊入口保留為轉接程式，原命令與參數仍可使用：

| 舊命令 | 正式命令 |
|---|---|
| `python examples/train_scenario_mix_ppo.py` | `python -m finrl.experiments.train_scenario_mix` |
| `python examples/log_training_rewards.py` | `python -m finrl.experiments.train_scenario_mix_logged` |
| `python examples/evaluate_scenario_mix.py` | `python -m finrl.experiments.evaluate_scenario_mix` |
| `python examples/evaluate_checkpoints.py` | `python -m finrl.experiments.evaluate_scenario_mix_checkpoints` |

彙整、比較、繪圖工具與研究 notebook 位於 [`analysis/scenario_mix/`](../../analysis/scenario_mix/README.md)，仍讀取相同輸出格式。`examples/` 分析腳本與原 notebook 路徑保留相容入口；共用的步數對齊邏輯仍由本套件的 `scenario_mix_reporting.py` 提供。

## 執行方式

一般訓練入口預設執行 A／B20／B50 × seeds 0／1／2，並以 validation final value 選模：

```bash
venv/bin/python -m finrl.experiments.train_scenario_mix \
  --prices ../TimeDiff/datasets/dow10/prices.csv \
  --synthetic ../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz \
  --out results/scenario_mix/new_run
```

記錄每個 episode 並保存所有 checkpoint；每個 process 執行一組 group／seed：

```bash
venv/bin/python -m finrl.experiments.train_scenario_mix_logged \
  --prices ../TimeDiff/datasets/dow10/prices_2004.csv \
  --synthetic ../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz \
  --group B90 --p 0.9 --seed 2 \
  --out results/scenario_mix/new_logged_run/B90_seed2
```

重新測試已選定模型。`--runs` 下每個子目錄須為單一 group／seed 的一般訓練輸出，包含 `config.json` 與 `<group>_seed<k>.zip`；這個入口不讀取 logged 入口的 `ckpt/` 格式：

```bash
venv/bin/python -m finrl.experiments.evaluate_scenario_mix \
  --runs results/scenario_mix/v1 \
  --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv \
  --test 2020-01-02 2025-12-31 \
  --out results/scenario_mix/new_evaluation
```

回測 logged 入口的全部 checkpoint；仍寫入 `--root/checkpoint_eval.csv`，若同名檔案存在仍會覆寫，與原入口相同：

```bash
venv/bin/python -m finrl.experiments.evaluate_scenario_mix_checkpoints \
  --root results/scenario_mix/new_logged_run \
  --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv
```

各入口加上 `--help` 可查看完整參數。

## 保留的實驗差異與產物

| 項目 | 一般訓練 | Logged 訓練 |
|---|---|---|
| 預設訓練期間 | 2009-04-01～2017-12-31 | 2010-01-01～2017-12-31 |
| 預設 validation | 2018-01-01～2019-12-31 | 2018-01-01～2019-12-31 |
| 執行單位 | 多組 group／seed | 單一 group／seed |
| 訓練成本 | `--commission` | `--train-commission`，省略時沿用 `--commission` |
| 模型保存 | validation 分數嚴格改善時保存最佳模型，同分保留較早者 | 保存每個 checkpoint，不自動選模 |
| 訓練後測試 | 預設 2020-01-02～2025-12-31，含基準與可選 crash 診斷 | 另呼叫 checkpoint 回測入口 |
| 主要產物 | `<group>_seed<k>.zip`、`*_valid.csv`、`*_test_daily.csv`、`summary.csv` 等 | `episodes.csv`、`valid.csv`、`ckpt/step_<t>.zip` |

兩種訓練維持原本的 500,000 名義步數、每 25,000 步呼叫一次 `learn(..., reset_num_timesteps=False)`，以及原 PPO／環境參數。PPO 完成整個 2,048 步 rollout，因此預設實際總步數仍是 532,480；檔名的名義 checkpoint ID 與 `actual_timesteps` 欄位保持原意。

股票順序、抽樣、seed、reward、TRF 成本、validation 分數公式、輸出欄位與既有覆寫檢查均維持原行為。Checkpoint 回測仍預設 15 workers、0.25% 費率及 GFC 額外視窗；指定 `--extra` 仍取代預設 GFC 視窗。

目錄整理沒有重跑、搬移或改寫歷史結果，也不會使舊 TRF 結果自動變成修正後結果。歷史結果的適用範圍與重跑需求見 [TRF 與 reward 步數修正說明](../../examples/README_scenario_mix_corrections.md)。

## 驗證

```bash
MPLCONFIGDIR=/private/tmp/scenario-mix-mpl venv/bin/python -m pytest \
  unit_tests/experiments/test_scenario_mix_entrypoints.py \
  unit_tests/experiments/test_scenario_mix_reporting.py \
  unit_tests/environments/test_portfolio_trf.py -q
```

入口測試涵蓋參數預設、共用 PPO 設定、選模分數、舊入口相容性，以及使用暫存資料與縮小訓練預算的訓練／保存／載入／回測流程。測試產物寫入 pytest 暫存目錄，不使用歷史結果目錄。
