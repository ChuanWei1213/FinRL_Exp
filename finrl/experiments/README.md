# Synthetic vs. Real PPO 實驗

這套 runner 用相同的 PPO pipeline 執行可設定的 studies。預設 `standard`
study 比較三種訓練資料組別：

- `real_trained`：只使用真實資料。
- `synthetic::<model>`：只使用該 generator model 的合成資料；若有多條 path，每條 path 等權抽樣。
- `real_synthetic::<model>`：真實與合成資料各占 50%；合成資料內各 path 等權抽樣。

實驗參數集中在 [`configs/synthetic_vs_real.json`](../../configs/synthetic_vs_real.json)。以下指令都假設目前位於專案根目錄。

## 1. 資料結構

真實資料維持一個 ticker 一個檔案：

```text
data/real/
├── AAPL.csv
├── AMGN.csv
└── ...
```

每個 CSV 至少需要 `Date, Open, High, Low, Close, Volume`。檔名（不含 `.csv`）必須與 ticker 相同。

合成資料使用一段完整時間序列，不再依實驗 window 分資料夾：

```text
data/synthetic/
└── <ticker_group>/
    └── <dataset_id>/
        ├── <model_a>/
        │   ├── path_001.csv
        │   └── path_002.csv
        └── <model_b>/
            └── path_001.csv
```

合成 CSV 需要 `date, tic, close, high, low`。同一個 dataset 內的所有 model/path 必須使用相同交易日與 ticker grid，而且每個 `(date, tic)` 只能有一筆。

新增資料集後，請同步更新 config 中的：

- `ticker_groups`：ticker group 與成分股。
- `active_ticker_groups`：沒有指定 `--ticker-group` 時預設執行的 group。
- `synthetic_datasets`：每個 ticker group 對應的 `dataset_id`。
- `studies.standard.synthetic_models`：standard study 要納入的 model 目錄名稱與顯示名稱。存在於資料夾、但未列在此處的 model 不會執行。

每個 study 都在 `studies` 內設定自己的 `training_groups`、`run_modes`，以及
`synthetic_models` 或 `synthetic_path_subsets`。舊的頂層
`synthetic_model_labels`、`run_modes` 格式不再接受。

`path-count` study 對同一 source model 取排序後的 nested prefixes，例如
`data[:10]`、`data[:50]`。如果某個 prefix 與 standard dataset 相同，可以用
`equivalent_source_models` 明示等價關係；runner 會逐 window 比對 ordered
canonical paths，完全相同才允許共用舊 cache。

目前預設為 `chosen_10`，資料 preflight 結果為 10 個真實資料檔、2 個 ARMD model、18 個 rolling windows；第一個 test 是 `2020-07-06` 至 `2020-10-01`，最後一個 test 是 `2024-10-07` 至 `2024-12-31`。

## 2. Walk-forward protocol

目前 config 的 rolling schedule 是：

```text
train: 504 trading days
validation: 126 trading days
test: 63 trading days
step: 63 trading days
```

最後不足 63 日的 test 仍會執行，並在輸出標記為 partial window。每個 window 都重新初始化並訓練一個 PPO，不會從上一個 window warm-start。

Portfolio initial state：

- Training：每個 episode 從全現金開始。
- Validation：從全現金開始。
- Independent test：每個 window 從全現金開始。
- Continuous test：只有第一個 window 從全現金開始；後續 window 承接上一個 window 的 final portfolio value、實際持倉權重與上一個 target action。跨 window 的第一次調倉會計入 fee 與 turnover。

因此，`independent` 適合比較單一市場環境；`continuous` 是主要的完整 OOS walk-forward 結果。Continuous chain 需要連續執行相鄰 windows；若只挑不相鄰的 `--window`，runner 會拒絕建立不連續的 chain。

## 3. 執行環境

如果尚未建立環境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

確認 CLI：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real --help
```

Server 上建議在 terminal 執行 runner；notebook 只負責載入結果與畫圖。長時間工作可放在 `tmux` 或其他 job scheduler 內，避免 SSH 中斷時一起停止。

## 4. 先列出 rolling windows

`--window` 使用的是完整 window name。可先用以下程式列出目前 config 會產生的 windows：

```bash
venv/bin/python - <<'PY'
from pathlib import Path

from finrl.experiments.synthetic_vs_real import load_experiment_config
from finrl.experiments.synthetic_vs_real import rolling_time_splits

root = Path.cwd()
config = load_experiment_config(root / "configs/synthetic_vs_real.json")

for ticker_group, tickers in config.select_ticker_groups(None):
    splits = rolling_time_splits(
        project_root=root,
        config=config,
        ticker_group=ticker_group,
        tickers=tickers,
    )
    for split in splits:
        print(
            ticker_group,
            split.name,
            f"train={split.train_start}:{split.train_end}",
            f"val={split.val_start}:{split.val_end}",
            f"test={split.test_start}:{split.test_end}",
        )
PY
```

日期區間使用 `[start, end)`；window name 中顯示的則是 test 的第一個與最後一個實際交易日。

## 5. 跑實驗

先跑 smoke。預設設定為 2 seeds、4,096 timesteps、含手續費：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode smoke \
  --stage all \
  --workers 4
```

建議第一次只跑第一個 window，確認整條 data → train → validation → independent/continuous test → artifacts 流程：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode smoke \
  --ticker-group chosen_10 \
  --window wf_504_126_63_000__test_20200706_20201001
```

跑完整設定。Standard full 預設為 1 seed、200,000 timesteps，同時執行
`no_fee` 與 `with_fee`：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode full \
  --stage all \
  --workers 4
```

執行 path-count ablation。每個 window 為 1 個 real-only 加上 5 個
Real+Synthetic PPO runs；smoke/full 都固定 seed `0` 與 fee `0.0025`：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --study path-count \
  --mode smoke \
  --stage all \
  --workers 4
```

目前設定要求 `nsdiff__500_paths` 至少包含 500 個 `path*.csv`；缺少時
preflight 會直接指出所需 source model 或 path 數量。

預設會訓練 study 中配置的所有 synthetic datasets。可用 include 或 exclude
限制單次執行，兩者不能同時使用；選擇只影響 synthetic variants，real-only
baseline 仍會保留：

```bash
# Standard：只訓練指定 models
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --study standard \
  --include-dataset nsdiff__50_paths \
  --include-dataset timediff__50_paths

# Path-count：source model ID 會選取該 source 的全部 counts
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --study path-count \
  --include-dataset nsdiff__500_paths

# 也可排除單一完整 variant ID
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --study path-count \
  --exclude-dataset nsdiff__500_paths__first_10_paths
```

`--include-dataset` 和 `--exclude-dataset` 都可重複指定。未知 ID 會在讀取或
訓練資料前直接報錯；CLI summary、experiment metadata 與 suite manifest 會
記錄 selection mode、selectors 和最後選中的 variants。

只跑特定 ticker group：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --mode smoke \
  --ticker-group chosen_10
```

`--ticker-group` 與 `--window` 都可以重複指定：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --mode smoke \
  --window wf_504_126_63_000__test_20200706_20201001 \
  --window wf_504_126_63_001__test_20201002_20201231
```

其他常用選項：

- `--stage all`：預設流程；先完成所有選定 window 的訓練，再統一執行 independent/continuous evaluation 與結果合併。
- `--stage train`：只建立 training cache，不執行 evaluation，也不更新 `latest_suite.json`。
- `--stage evaluate`：要求所有 training artifacts 已存在，只執行 evaluation 與合併；缺少模型時會列出 run IDs 後停止。
- `--workers <N>`：同時執行的 PPO process 數量，預設 1；每個 worker 限制為單一 CPU thread。可用 `--workers 1` 進行循序除錯。
- `--include-dataset <id>`：只保留指定 variant 或 source model；可重複指定。
- `--exclude-dataset <id>`：排除指定 variant 或 source model；可重複指定，不能與 include 同時使用。
- `--quiet`：隱藏 tqdm progress bar，保留狀態訊息；適合寫入 server log。
- `--force-retrain`：在 `all/train` 重新訓練；同時略過 evaluation cache。搭配 `evaluate` 時保留模型、只強制重算 evaluation。Fingerprint cache 仍可使用，因為它不包含模型訓練結果。
- `--no-cache`：不讀寫 training、evaluation 或 persistent fingerprint cache；同一 process 內仍會去重 source SHA 與 fingerprint 工作。模型寫到該實驗的 `uncached_runs/`，正式結果檔照常產生。

若要明確分兩階段執行：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode full \
  --stage train \
  --workers 4

venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode full \
  --stage evaluate
```

平行訓練的 tqdm 由主程序顯示完成的 PPO runs；worker 不各自輸出 progress bar。
需要 materialize 資料時，runner 會依序顯示讀取/驗證 synthetic paths、切割並
檢查 date grids、計算 per-path scaler maxima，以及轉換 subset variants 的
progress bars。Evaluation 另有每個 window 的 independent bar，以及 suite
層級的 continuous-chain bar，postfix 會顯示 cache hits 與實際執行數。

若使用 `tmux`：

```bash
tmux new -s finrl-svr
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode full \
  --quiet
```

按 `Ctrl-b`、再按 `d` 可離開 session；用 `tmux attach -t finrl-svr` 回到實驗。

## 6. Train cache

共用 cache 位於：

```text
results/ppo_synthetic_vs_real/train_cache/
```

新 cache key 只依賴會影響訓練結果的語意內容，包括 train/validation 日期、
ordered source-path canonical rows、training group 類型、PPO/environment 設定、
fee 與 seed。Study、model 名稱、檔名與來源目錄不在 key 內，因此
`nsdiff__50_paths` 與經驗證等價的 `nsdiff__500_paths[:50]` 可以共用模型。
Path 的數量與順序仍在 fingerprint 中，因為 seeded episode sampling 會受 path
index 影響。

Runner 會先找新的 semantic key；找不到時，standard runs 與配置
`equivalent_source_models` 的 path-count variants 會再計算舊版
path-sensitive key。舊 entry 命中後直接使用原本的模型與 logs，不會複製或
改寫舊 cache。Legacy 命中時，預期的 semantic cache 目錄只會新增一個小型
`semantic_alias.json`，指向原本的 `status.json`；下一次可直接循 alias 命中，
不必重算 legacy fingerprint。Alias target 未完成或缺少模型/logs 時會被忽略。

Fingerprint 的持久化衍生 cache 位於：

```text
results/ppo_synthetic_vs_real/fingerprint_cache/v1/
```

每次 suite 都會對每個 source file 完整計算一次 raw SHA-256，並在 hash 前後
檢查檔案狀態；同一 suite 中途改檔會立即失敗。SHA、ordered source topology
與 train/validation 日期共同定位已算過的 semantic/legacy fingerprint，因此
warm run 不需再次做 Decimal canonical CSV parsing。單一 process 內，同一檔案
即使被多個 windows、training groups 或 nested path prefixes 使用，也只計算
一次 raw SHA；nested prefixes 會共用共同前綴的 canonical parsing。

Training 第一輪先建立輕量 identity 並檢查所有 runs。若 `--stage train` 全部
cache hit，且既有 `data_summary.csv`、`scale_summary.csv` 完整，就不讀 synthetic
dataframe、不 fit scaler、也不做 transforms；缺 summary 時才 materialize 重建。
`--stage all` 仍保留全 windows training barrier，之後 evaluation 每個 window
只 materialize 一次。現有 `train_cache/*/status.json` 在一個 suite 中最多完整
掃描一次。

Test 日期、evaluation mode 和 continuous test 的初始持倉不在 training key
內，因此相同訓練條件可安全重用模型。

每個完成的 cache entry 會包含：

```text
best_model.zip
last_model.zip
training_log.csv
validation_log.csv
status.json
```

正常重跑時，runner 會先顯示 cache hit 與需要訓練的數量。資料內容或訓練 config 改變後，fingerprint/key 會改變，不需要手動清空舊 cache。

每個 window 的 `preparation_timings.csv` 會記錄 discovery、raw SHA、semantic /
legacy fingerprint hits 與 computations、training-cache lookup、dataframe
materialization、scaler/transform 和總耗時。CLI summary 與 suite manifest 的
`fingerprint_statistics` 則提供 suite 累計數字。`fingerprint_cache/v1/` 可安全
刪除；它不含模型或正式實驗 artifacts，下一次執行會自動重建。

## 6.1 Evaluation cache

Evaluation cache 位於：

```text
results/ppo_synthetic_vs_real/evaluation_cache/
├── independent/
└── continuous/
```

- Independent evaluation 以單一 window 的 PPO run 為 cache 單位；Buy & Hold 則以 window × fee 為單位。
- Continuous evaluation 以完整 `(ticker_group, group, fee, seed)` test chain 為單位，chain 內仍依時間順序承接 portfolio value、actual weights 與 last action。
- Semantic cache key 包含模型內容、實際 evaluation frames、日期、fee、seed 與 evaluation environment，不包含 study、group ID 或顯示名稱。
- Independent 與 continuous evaluation 都會 fallback 至經驗證等價 run 的舊 standard cache；載入後只把 group、model 與 label 改成目前 variant，數值結果不變。
- 每個 entry 的 `status.json` 必須為 `completed: true`，且所有 CSV artifacts 都存在且可讀，才會視為 cache hit；中斷或損壞的 entry 會自動重算。
- 每個 window 會輸出 `evaluation_timings.csv`，suite 另有 `continuous_evaluation_timings.csv`，記錄 cache hit 與耗時。

## 7. 輸出結果

每次 suite 完成後，CLI 最後會輸出 JSON，包含 `manifest` 與 `suite_root`。最新一次 suite 的指標檔可從以下 pointer 找到：

```text
results/ppo_synthetic_vs_real/<smoke|full>/latest_suite.json
```

Path-count 使用相同 artifact 格式，但放在獨立目錄方便同一 source model 的
path counts 比較：

```text
results/ppo_synthetic_vs_real/path-count/<smoke|full>/latest_suite.json
```

兩個 studies 共用頂層的 `train_cache/` 與 `evaluation_cache/`。Final suite
CSV/JSON 不直接複製舊 study；runner 會從共用 cache 重新組裝，確保 `study`、
`synthetic_source_model`、`synthetic_path_count` 與 cache provenance 正確。

Suite 層級的主要輸出：

```text
manifest.json
test_metrics.csv
aggregate_summary.csv
paired_deltas.csv
continuous_metrics.csv
continuous_window_metrics.csv
continuous_daily_returns.csv
continuous_equity_curves.csv
continuous_portfolio_weights.csv
window_transition_log.csv
```

- `test_metrics.csv`：每個 independent test window、group、fee、seed 的 cumulative return、CAGR、Sharpe、annualized volatility、MDD、turnover 等指標。目前 runner 尚未輸出 Sortino 與 Calmar。
- `paired_deltas.csv`：相同 window/fee/seed 下，synthetic-only 或 real+synthetic 相對 real-only 的 paired 差異。
- `continuous_metrics.csv`：各 group/fee/seed 串接完整 OOS test chain 後的總體指標。
- `window_transition_log.csv`：continuous test 的 window 邊界、承接資產與持倉資訊，可用來檢查 state continuity。

每個 window 的 experiment 目錄另有：

```text
run_table.csv
ppo_period_metrics.csv
backtest_results.csv
equity_curves.csv
portfolio_weights.csv
median_sharpe_representatives.csv
model_references.json
evaluation_timings.csv
```

## 8. 在 notebook 載入結果

不要在 notebook 重新訓練。先在 terminal 跑完，再載入 `latest_suite.json`：

```python
from finrl.experiments.synthetic_vs_real_plots import load_suite_artifacts
from finrl.experiments.synthetic_vs_real_plots import plot_continuous_equity_curves
from finrl.experiments.synthetic_vs_real_plots import plot_test_metric_by_window

artifacts = load_suite_artifacts(
    "results/ppo_synthetic_vs_real/smoke/latest_suite.json"
)

display(artifacts.continuous_metrics)
plot_test_metric_by_window(
    artifacts,
    metric="sharpe",
    ticker_group="chosen_10",
    commission_name="with_fee",
)
plot_continuous_equity_curves(
    artifacts,
    ticker_group="chosen_10",
    commission_name="with_fee",
)
```

其他現成圖表函式位於 [`synthetic_vs_real_plots.py`](synthetic_vs_real_plots.py)：learning curves、paired metric deltas、單一 window 的 representative equity curve 與 portfolio weights。

## 9. 常見錯誤

- `No module named 'gym'`：請先確認 server 已同步最新版；portfolio optimization base environment 現在會在 legacy Gym 缺少時自動 fallback 至 Gymnasium。完整 FinRL 環境仍可執行 `venv/bin/python -m pip install "gym==0.26.2"`，重新依 `requirements.txt` 建立環境也會安裝此版本。
- `No module named 'alpaca_trade_api'`：Synthetic-vs-Real runner 不使用 Alpaca；目前 `finrl` 已將 paper-trading entry point 改為 lazy import。請確認 server 使用包含此修改的版本，並從專案根目錄執行。只有另外使用 Alpaca paper trading 時才需要執行 `pip install "alpaca_trade_api>=2.1.0"`。
- `Synthetic dataset directory not found`：`synthetic_datasets[ticker_group]` 與實際 `dataset_id` 目錄不一致。
- `No synthetic paths found`：model 下沒有符合 `path*.csv` 的檔案。
- `expected tickers ... got ...`：ticker group 與 synthetic CSV 中的 ticker 集合不一致。
- `incomplete ticker grid` 或 duplicate row：同一天缺少 ticker，或 `(date, tic)` 重複。
- 真實資料日期不足：real data 必須完整覆蓋每個 window 的 train、validation 與 test。
- Continuous chain 不連續：選取的 windows 中間有缺口。只分析零散 windows 時，可使用 independent outputs；要產生 continuous 結果則選相鄰 windows 或直接跑全部 windows。
