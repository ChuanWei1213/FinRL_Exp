# scenario_mix：TRF 與 reward 步數修正

本文件適用於 `results/scenario_mix/` 的 v2、v3 歷史結果。修正程式不會讓既有 CSV、模型或圖表自動變成修正後的實驗結果；舊結果應保留供稽核，新實驗使用另一個輸出目錄。

訓練與回測的正式入口已移至 `finrl.experiments`，舊 `examples/` 命令仍相容；對照表與完整用法見 [Scenario-mix 實驗入口](../finrl/experiments/README_scenario_mix.md)。此次入口整理不改變本文件記錄的修正、實驗設定或歷史結果。

彙整、比較與繪圖工具現位於 [`analysis/scenario_mix/`](../analysis/scenario_mix/README.md)，下文的舊腳本名稱仍可使用。搬移不改變指標、步數對齊、選模規則或歷史結果的適用範圍。

## 修正內容

### TRF 成本

`env_portfolio_optimization.py` 的 TRF 分子改用交易前、已經歷價格漂移的現金權重 `last_weights[0]`，分母仍使用目標現金權重 `weights[0]`。公式依據 [Jiang et al. (2017), Eq. (14)](https://arxiv.org/html/1706.10059v2)。買入費率沿用原論文對「支出現金」扣費的慣例，沒有改成另一種券商計費定義。

在價格不動、費率 0.25% 時，預期結果為：

| 操作 | 修正前損失 | 修正後損失 |
|---|---:|---:|
| 全現金 → 全股票 | 0% | 0.25% |
| 全股票 → 全現金 | 約 0.500627% | 0.25% |
| 股票 A → 股票 B | 0.499375% | 0.499375% |
| 維持交易前權重 | 0% | 0% |

費率為零時 `mu = 1`，修正不改變環境的 observation、reward 或淨值。初始等權持股仍在 reset 直接建立；此處沒有改初始建倉機制。

TRF 迭代使用 float64 計算，以符合原本 `1e-10` 的停止條件；動作與 observation 的 dtype 不變，避免 NumPy 2 將金融成本計算降成 float32。

### 步數與 reward 分桶

兩個訓練入口保留原本的 `learn(eval_freq, reset_num_timesteps=False)` 呼叫次數與訓練排程，避免僅為修正紀錄而改變實驗預算或模型。PPO 會完成整個 rollout：

```text
每次實際步數 = ceil(25,000 / 2,048) × 2,048 = 26,624
20 次 learn 的實際總步數 = 532,480
```

- `timesteps`：保留名義 checkpoint ID，例如 `500000`，用於檔名及資料連接。
- `actual_timesteps`：直接記錄 `model.num_timesteps`，用於繪圖與 episode 分桶。
- `rollout_steps`：新的 reward logging 設定記錄 `model.n_steps * model.n_envs`。
- 舊 v2/v3 沒有實際步數欄位時，報表工具依歷史單環境、2,048 步 rollout 與 `eval_freq` 重建；缺漏或不規則的歷史排程會報錯，不直接猜測。
- reward 以 episode **結束時點**分配到 `(上一個實際 checkpoint, 目前實際 checkpoint]`。跨界 episode 不拆分，不能將區間平均當成單一固定策略的評估。
- 測試指標也連接到相同的實際步數，避免只修 reward 軸而讓 validation/test 軸繼續錯位。

`plot_reward_curves.py` 的 `picked_steps.csv` 同時提供名義 `_step` 與 `_actual_step`；`evaluate_checkpoints.py` 直接由載入的模型讀取實際步數。

## 哪些結果需要重做

| 結果 | 重訓 | 重新驗證、選模 | 重新測試 | 重畫圖／重建表 |
|---|---|---|---|---|
| `v2_rewardlog` | **需要**，15 組訓練成本皆為 0.25% | 需要 | 需要 | 需要 |
| `v2_sharpe` | **需要**，或從設定相同的新版 v2 訓練選出模型 | 需要 | 需要 | 需要 |
| `v2_sharpe_test2025` | 本身是延長測試，沿用新版 v2 模型 | 沿用新版選模結果 | **需要** | 需要 |
| `v3_nocost` | **不需要**，保留所有既有 checkpoint | **需要**，評估費率仍為 0.25% | **需要** | 需要 |
| 舊 `episodes.csv` 的時間對齊 | 不需要 | 不需要 | 不需要 | **需要** |
| Buy-and-hold 基準 | 不需要 | 不適用 | 此次修正不需要 | 比較報表更新時可沿用 |

### v2

錯誤 TRF 成本已進入 PPO 的 reward、advantage 與參數更新。只用新環境回測舊模型，可以衡量「舊策略在正確成本下的表現」，不能代表「正確成本訓練的模型」。要重新比較有成本訓練與零成本訓練，A/B20/B50/B70/B90 × seeds 0/1/2 都需要從頭訓練。

修正後一次保存全部 checkpoint，即可分別依 final value 或 Sharpe 選模，並測試至 2022 或 2025；設定相同時不需要為每個選模規則或測試終點重訓一遍。

舊 v2 的 episode reward 可以重新分桶，用於描述舊實驗；僅有 episode 總 reward 不能還原正確成本下的訓練歷程，也不能用它代替重訓。

### v3

`train_commission=0` 使 TRF 修正不影響訓練更新。且訓練是固定步數、沒有依驗證結果調整學習或提前停止，因此原有 checkpoint 可繼續使用。

1. 對 15 組模型的全部 20 個 checkpoint 重跑 2018–2019 validation，共 300 個，重建 final value、reward、Sharpe、Calmar 與選模結果。
2. 對新選出的 checkpoint 及訓練終點，重跑 2020–2025 與 GFC 額外視窗。若保留「每個 checkpoint 的測試曲線」，則全部 300 個 checkpoint 都需要重測。
3. 重新建立 `checkpoint_eval.csv`、`picked_table.csv`、`picked_steps.csv` 與比較圖。單獨重跑 `evaluate_checkpoints.py` 不會重建 `valid.csv`，不能仍拿舊 validation 選模。
4. v3 的 `episodes.csv` reward 本身可保留，只需由修正後的報表工具重新分桶。

同一個固定 checkpoint 的 observation 只含價格與上一個目標 action；TRF 的共同淨值倍率會在權重正規化時抵消。因此它的持股／換手路徑理論上不因成本修正而變（浮點誤差除外），但淨值、報酬、回撤、Sharpe 必須重算。**重新選出的 checkpoint 可能不同，所以選模後的持股／換手比較也必須更新。**

Buy-and-hold 使用獨立的解析式，沒有呼叫這段 TRF；價格資料、合成路徑、episode 來源與抽樣比例也不因本次修正失效。

## 重跑前的資料需求

初次修正時，本地 v2/v3 logging 目錄缺少 checkpoint 與原始資料。2026-10-11 已由 `257bf771` 補齊 600 個 `ckpt/step_*.zip`，並補回兩份價格 CSV 與 `paths.npz`；後續新舊 TRF 對照另存於 `results/scenario_mix/trf_ablation_20261011/`，入口見 [分析工具](../analysis/scenario_mix/README.md#trf-修正前後的成對實驗)。v3 現有模型可直接重新驗證與測試，無須因資料曾缺少而重訓。

`8fa237e3` 程式修正本身保留既有歷史結果；後續實驗也使用獨立目錄，不覆寫原始紀錄。

2026-10-12 已完成 600 個舊 checkpoint 重測、新舊 TRF 各 15 組完整重訓與驗證；[結果與解讀](../results/scenario_mix/trf_ablation_20261011/report/analysis.md)分開呈現成本重算、重新選模與學習結果的影響。

## 驗證

`unit_tests/environments/test_portfolio_trf.py` 涵蓋現金買入、清倉、換股、不交易、部分配置、價格漂移、零費率，以及隨機配置的獨立現金收支檢查；同時涵蓋一般與 scenario-mix fast 環境。

`unit_tests/experiments/test_scenario_mix_reporting.py` 涵蓋 rollout 邊界、舊紀錄重建、尾端超過名義總步數的 episode、不規則紀錄、測試表連接，以及小型真實 PPO 訓練。

歷史資料檢查涵蓋 30 組紀錄的 507,328 個 episode，包含原本超過名義 500,000 步而被漏掉的 30,856 個 episode；450 筆既有選模指標在單純修正時間對齊後保持一致。這項檢查只驗證紀錄與報表，不代表舊成本績效已重新回測。
