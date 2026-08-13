# Synthetic vs. Real PPO 實驗

這套 runner 用相同的 PPO 設定，比較三種訓練資料組別：

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
- `synthetic_model_labels`：要納入實驗的 model 目錄名稱與圖表顯示名稱。存在於資料夾、但未列在此處的 model 不會執行。

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
  --mode smoke
```

建議第一次只跑第一個 window，確認整條 data → train → validation → independent/continuous test → artifacts 流程：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode smoke \
  --ticker-group chosen_10 \
  --window wf_504_126_63_000__test_20200706_20201001
```

跑完整設定。預設為 3 seeds、200,000 timesteps，同時執行 `no_fee` 與 `with_fee`：

```bash
venv/bin/python -m finrl.experiments.run_synthetic_vs_real \
  --config configs/synthetic_vs_real.json \
  --mode full
```

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

- `--quiet`：隱藏 tqdm progress bar，保留狀態訊息；適合寫入 server log。
- `--force-retrain`：忽略相容的 train cache，重新訓練並覆寫該 cache entry。
- `--no-cache`：完全不讀寫共用 train cache，模型寫到該實驗的 `uncached_runs/`。

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

Cache key 只依賴會影響訓練結果的內容，包括 train/validation 日期、實際訓練資料 fingerprint、ticker group、資料組別、PPO/environment 設定、fee 與 seed。Test 日期、evaluation mode 和 continuous test 的初始持倉不在 key 內，因此相同訓練條件可安全重用模型。

每個完成的 cache entry 會包含：

```text
best_model.zip
last_model.zip
training_log.csv
validation_log.csv
status.json
```

正常重跑時，runner 會先顯示 cache hit 與需要訓練的數量。資料內容或訓練 config 改變後，fingerprint/key 會改變，不需要手動清空舊 cache。

## 7. 輸出結果

每次 suite 完成後，CLI 最後會輸出 JSON，包含 `manifest` 與 `suite_root`。最新一次 suite 的指標檔可從以下 pointer 找到：

```text
results/ppo_synthetic_vs_real/<smoke|full>/latest_suite.json
```

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
