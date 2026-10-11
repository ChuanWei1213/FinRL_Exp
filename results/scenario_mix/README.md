# Scenario-mix 各版本實驗總覽

更新日期：2026-10-12。歷史版本依據 `TimeDiff-mix` 的程式、各 run 的 `config.json`、CSV 與本地模型檔盤點；`257bf771` 補齊模型後，已另行完成 TRF 對照實驗。

歷史版本共有 **6 個訓練／回測結果目錄、2 個比較報表目錄**。以下記錄實際執行設定，不以目前 CLI 預設值代替歷史設定。這些舊目錄仍保留 **TRF 修正前**的績效；修正後的新一輪結果另存於 [trf_ablation_20261011](trf_ablation_20261011/report/analysis.md)，包含 600 個舊 checkpoint 重測與 30 組完整成對重訓。

## 1. 版本與研究問題

| 目錄 | 類型／研究問題 | 訓練費率 | 驗證／測試費率 | 選模或比較規則 | 測試範圍 |
|---|---|---:|---:|---|---|
| [v1](v1/) | 基礎實驗：增加合成暴跌情境比例，是否改變持股及抗跌能力？ | 0.25% | 0.25% | 2018–2019 validation final value 最高 | 2020–2022 |
| [v1_test2025](v1_test2025/) | 將 v1 已選定模型的回測延長 | 沿用 v1 | 0.25% | 沿用 v1，不重新選模 | 2020–2025 |
| [v2_sharpe](v2_sharpe/) | 將選模依據改為 validation Sharpe | 0.25% | 0.25% | 2018–2019 validation Sharpe 最高 | 2020–2022 |
| [v2_sharpe_test2025](v2_sharpe_test2025/) | 將 v2 Sharpe 已選定模型的回測延長 | 沿用 v2 Sharpe | 0.25% | 沿用 v2 Sharpe，不重新選模 | 2020–2025 |
| [v2_rewardlog](v2_rewardlog/) | 重跑有成本訓練，記錄每個 episode 與全部 checkpoint，觀察學習過程 | 0.25% | 0.25% | 保留全部 checkpoint，事後比較 final value／Sharpe／訓練終點 | 2020–2025、GFC |
| [v3_nocost](v3_nocost/) | 移除訓練交易成本，檢查成本對學習與持股的影響 | **0%** | **0.25%** | 同 v2 reward logging | 2020–2025、GFC |
| [compare_selection](compare_selection/) | 比較 final value 與 Sharpe 選模 | 來源為 v1、v2 Sharpe | 沿用來源結果 | 每個 seed 各自選模後，再彙整組別 | 使用延長至 2025 的結果 |
| [compare_train_cost](compare_train_cost/) | 比較有成本與零成本訓練 | 來源為 v2 reward logging、v3 | 沿用來源結果 | 各自比較三種 checkpoint 選取規則 | 2020、2022、2025、2020–2025、GFC |

`v2` 並非單一目錄：`v2_sharpe` 改變選模方式；`v2_rewardlog` 提供完整訓練軌跡。兩個 `*_test2025` 都是延長回測，不是新增訓練方法；兩個 `compare_*` 都是衍生報表。

回測與比較報表的來源關係如下：

```text
v1 ──延長回測──> v1_test2025 ──────────────────┐
v2_sharpe ──延長回測──> v2_sharpe_test2025 ────┴─> compare_selection

v2_rewardlog（訓練成本 0.25%）──┐
v3_nocost（訓練成本 0%）────────┴─> compare_train_cost
```

有訓練的版本是從頭執行，不是載入上一版模型接續訓練。

## 2. 共用實驗設計

### 資料、資產與期間

| 項目 | 已執行實驗的設定 |
|---|---|
| 股票順序 | AAPL、AMGN、CRM、CSCO、IBM、INTC、MSFT、NKE、VZ、WMT |
| 配置 | 10 檔股票＋現金，多頭、無槓桿 |
| 初始資金 | 100,000 |
| 初始權重 | 現金與 10 檔股票各 1/11，股票合計約 90.91% |
| FinRL 訓練期間 | **2010-01-01～2017-12-31**，所有已保存設定一致 |
| Validation | 2018-01-01～2019-12-31 |
| 第一輪測試 | v1、v2 Sharpe 設定為 2020-01-01～2022-12-31，實際依交易日資料 |
| 延長測試 | 兩個 `*_test2025` 設定為 2020-01-02～2025-12-31 |
| 歷史價格設定路徑 | `../TimeDiff/datasets/dow10/prices_2004.csv` |
| 延長回測價格設定路徑 | `../TimeDiff/datasets/dow10/prices_2004_2025.csv` |
| 合成路徑設定位置 | `../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz` |

以上資料路徑以專案根目錄為工作目錄解讀。目前一般訓練 CLI 預設起點為 2009-04-01，不能據此將歷史實驗寫成從 2009 年訓練。

v2 reward logging／v3 的 `checkpoint_eval.csv` 保存 2020～2025 各年、2020–2025 全期及 `gfc`。原訓練 `config.json` 沒有保存後續回測的完整日期參數；目前 checkpoint 回測入口預設主視窗為 2020-01-02～2025-12-31、GFC 為 2008-01-02～2009-06-30。CSV 可確認期間標籤，精確起迄日仍需搭配原執行命令核實。

### 混合方式與組別

每個 episode 先使用 50 日觀察資料，再交易 `k` 日真實行情與 21 日後續行情；`k` 在 0～21 均勻抽樣。後續 21 日依機率 `p` 選擇合成路徑，否則使用真實行情。政策輸入不包含「真實／合成」或 scenario 編號。

| 組別 | 合成 episode 機率 `p` | 合成交易日占比的理論近似值 |
|---|---:|---:|
| A | 0% | 0% |
| B20 | 20% | 13.33% |
| B50 | 50% | 33.33% |
| B70 | 70% | 46.67% |
| B90 | 90% | 60.00% |

第三欄是 `p × 21 / (E[k] + 21)`，其中 `E[k] = 10.5`，不是實測比例。**B90 表示 90% episode 採用合成後段，不表示 90% 訓練交易日都是暴跌資料。** 每個版本均有上述 5 組 × seeds 0、1、2，共 15 個 run。

### TimeDiff 暴跌情境

下表取自 [第一輪研究 notebook](../../analysis/scenario_mix/scenario_mix_v1_summary.ipynb) 保存的生成紀錄，**本次沒有重新讀取原始 `paths.npz` 驗算**。TimeDiff 訓練期間記錄為 2005-01～2017-12，包含 2008 年；供 FinRL 使用的生成錨點落在 2010-01～2017-12。

| 情境 | 目標 21 日報酬／年化波動 | 實際採用的生成條件：報酬／波動 | 記錄的平均實現報酬／年化波動 | 路徑數 |
|---|---|---|---|---:|
| S1／scenario 0 | −10%／40% | −10%／50% | −9.51%／38.73% | 2,013 |
| S2／scenario 1 | −20%／60% | −23%／100% | −20.15%／61.36% | 2,013 |
| S3／scenario 2 | −30%／80% | −36%／140% | −29.95%／78.44% | 2,013 |

合計 6,039 條、每條 21 日的 10 檔股票路徑。表中報酬與波動描述的是 10 檔股票每日等權再平衡的情境投資組合，並非 FinRL 策略績效。GFC 雖早於 FinRL 訓練期間，TimeDiff 已看過 2008 年，因此不能將 GFC 稱為整套流程完全未見過的樣本外資料。

### 政策、reward 與訓練預算

| 項目 | 設定 |
|---|---|
| Observation | 過去 50 日 close／high／low，按各股票最新 close 正規化，加上上一個目標 action |
| 政策 | PPO `MultiInputPolicy`＋`EIIEExtractor` |
| 特徵提取器 | `k_size=3`、`conv_mid=2`、`conv_final=20` |
| 動作 | 11 維配置；`action_scale=5`，經 softmax 轉為權重 |
| Reward | 每步 `100 × log(V_t / V_{t-1})`；episode 加總等於 `100 × log(V_end / V_start)` |
| PPO | learning rate `3e-4`、`n_steps=2048`、batch size `256`、`n_epochs=10` |
| 折現／GAE | `gamma=0.99`、`gae_lambda=0.95` |
| 探索參數 | `log_std_init=-2`、`ent_coef=0` |
| 執行 | CPU、單一環境、Torch threads 1 |
| 名義預算 | 每 25,000 步驗證一次，共 500,000 步、20 個 checkpoint |
| 實際預算 | 每次 learn 完成 26,624 步，共 **532,480 步** |

v1 較早的 A／B20／B50 設定檔沒有寫出探索參數與選模規則；上表已對照當時程式，為 `log_std_init=-2`、`ent_coef=0`、final value 選模。B70／B90 的設定檔則有明列。

Reward 沒有額外的暴跌、持股或回撤懲罰。改成 Sharpe 選模只改變最後保留哪個 checkpoint，並沒有把 PPO 的訓練目標改成 Sharpe。

**初始免建倉費與 v3 零成本訓練是不同設定。** 各版環境在 reset 直接建立初始等權配置，不透過第一次買入交易扣費；之後是否收費由訓練費率決定。v3 的 `train_commission=0` 使整段訓練的買入、賣出與再平衡都不收費，不只是免初始建倉費，也不是只把買入費率設為零。

## 3. 各版本的內容與產物

### v1：混合比例基礎實驗

先執行 A／B20／B50，之後加入 B70／B90。各 seed 以 validation final value 選出最佳 checkpoint，再於 2020–2022 連續回測，觀察合成比例與報酬、回撤、持股的關係。

- 每個 run 有設定、validation 軌跡、所選模型 `.zip`、測試每日淨值／權重、兩種 buy-and-hold 基準及摘要。
- 本地共 15 個所選模型，檔案格式為 `v1/<group>_seed<n>/<group>_seed<n>.zip`。
- 根目錄 [all_runs.csv](v1/all_runs.csv) 有 68 筆：15 個策略 run＋2 個基準，各 4 個期間（2020、2021、2022、2020–2022）。
- [validation_curves.csv](v1/validation_curves.csv) 有 300 筆，即 15 run × 20 checkpoint。
- Crash response 診斷目前只有 **B70、B90**，共 6 run × 4 種來源（real／3 個 scenario）＝24 筆，不能推定 A／B20／B50 也有同樣紀錄。

### v1_test2025：延長 v1 的測試期間

沿用 v1 的最佳模型，不重訓、不重新選模。`config.json` 的 `evaluated_from` 指回 v1 模型，另以延長價格資料回測 2020–2025。

- [all_runs.csv](v1_test2025/all_runs.csv) 有 119 筆：17 個策略／基準 × 7 個期間（6 個年度＋全期）。
- 本目錄沒有獨立模型 `.zip`；保存的 validation CSV 為沿用紀錄，不代表重新執行驗證。
- 沒有新增 crash response 診斷。此目錄適合比較同一策略在更長期間的表現，不能當成新策略。

### v2_sharpe：改用 validation Sharpe 選模

資料、混合比例、PPO、seed、成本與訓練預算沿用 v1。只將最佳模型判定改為 2018–2019 validation Sharpe；同分保留較早的 checkpoint。

- 本地有 15 個所選模型；主要輸出形式與 v1 相同，測試期間仍為 2020–2022。
- [all_runs.csv](v2_sharpe/all_runs.csv) 有 68 筆；[validation_curves.csv](v2_sharpe/validation_curves.csv) 有 300 筆。
- Crash response 診斷涵蓋全部 15 run，共 60 筆。
- 已逐一核對 v1 與本版本的 15 run × 20 checkpoint：對應的 validation final value **差異全為零**。現有紀錄支持「同一條觀測到的訓練／驗證軌跡，採不同選模規則」；此核對不等於逐位元組比較模型參數。

### v2_sharpe_test2025：延長 Sharpe 所選模型的測試期間

沿用 v2 Sharpe 的所選模型至 2025 年底，`evaluated_from` 指向 `v2_sharpe`。沒有新增訓練、重新選模或 crash response 診斷；本地沒有另存模型 `.zip`。

[all_runs.csv](v2_sharpe_test2025/all_runs.csv) 同樣有 119 筆。與 `v1_test2025` 比較，才能在相同測試期間評估兩種選模規則。

### v2_rewardlog：完整記錄有成本訓練

重新執行相同的有成本訓練，記錄每個 episode 的 reward、長度、來源、scenario 與 `k`，並在每個驗證點保存 checkpoint，供事後研究學習曲線及選模。

- 每個 run 有 `config.json`、`episodes.csv`、`valid.csv`。15 run 合計 **253,664 個 episode**；20 個 checkpoint × 15 run＝300 個驗證點。
- `valid.csv` 記錄 final value、reward、Sharpe、Calmar 與平均持股；記錄 Calmar 不代表已有 Calmar 選模的獨立實驗。
- [checkpoint_eval.csv](v2_rewardlog/checkpoint_eval.csv) 有 **2,400 筆**：15 run × 20 checkpoint × 8 期間（2020～2025、全期、GFC）。
- 另有 [picked_steps.csv](v2_rewardlog/picked_steps.csv) 與 `fig_reward_vs_steps.png`，記錄 final value／Sharpe 所選步數與訓練曲線。
- 與 v1 對應的 300 個 validation final value 也全部一致；新增 logging 沒有在這些紀錄中形成另一種訓練效果。
- 訓練程式會保存 `ckpt/step_<名義步數>.zip`，但**目前本地目錄沒有這些 checkpoint 檔案**；也沒有逐 checkpoint 的每日測試權重／淨值檔，只有彙總績效。

### v3_nocost：移除訓練成本

承接 v2 reward logging 的實驗設計，主要變更為 `train_commission=0.0`；validation 與 test 仍用 `commission=0.0025`。用來檢查交易成本是否影響 PPO 的學習、持股或換手。

- 仍為 5 組 × 3 seeds × 20 checkpoint；15 run 合計 **253,664 個 episode**。
- [checkpoint_eval.csv](v3_nocost/checkpoint_eval.csv) 同樣有 2,400 筆、8 個期間。
- 訓練不是直接最小化驗證 Sharpe 或測試回撤；成本改變的是訓練 reward，選模規則仍在訓練後比較。
- 本地同樣缺少 `ckpt/step_*.zip`，也沒有逐 checkpoint 的每日測試路徑。
- 本目錄未另存與 v2 同名的 reward 圖／picked steps；兩版本比較主要在 `compare_train_cost`。

## 4. 比較報表涵蓋什麼

### compare_selection：final value 與 Sharpe

| 報表標籤 | Validation／訓練來源 | 測試來源 |
|---|---|---|
| `final_value` | `v1` | `v1_test2025` |
| `sharpe` | `v2_sharpe` | `v2_sharpe_test2025` |

- [compare_table.csv](compare_selection/compare_table.csv)：98 筆，5 個策略組＋2 個基準 × 2 個標籤 × 7 個期間，含平均績效與相對 final value 選模的差異。
- [selected_checkpoints.csv](compare_selection/selected_checkpoints.csv)：30 筆，即 15 run × 2 種選模規則。
- [training_identical.csv](compare_selection/training_identical.csv)：15 筆，對應 run 的 validation final value 最大差異皆為零。
- [crash_response_compare.csv](compare_selection/crash_response_compare.csv)：28 筆；final value 側只有 B70／B90，Sharpe 側有全部組別，**診斷覆蓋不對稱**。
- `fig_compare.png` 為上述比較的視覺化，不是新增訓練。

### compare_train_cost：有成本與零成本訓練

`train_cost` 來自 `v2_rewardlog`，`train_nocost` 來自 `v3_nocost`。每個 run 各自比較：

1. `pick_final_value`：validation final value 最高的 checkpoint。
2. `pick_sharpe`：validation Sharpe 最高的 checkpoint。
3. `last_step`：名義 500,000 步的最終 checkpoint，對應實際 532,480 步。

[picked_table.csv](compare_train_cost/picked_table.csv) 有 **450 筆**：2 個訓練版本 × 3 種規則 × 15 run × 5 個期間。表內期間為 2020、2022、2025、2020–2025、GFC；原始 checkpoint 評估另有 2021、2023、2024，沒有顯示在此選模表中。

`fig_steps.png` 比較真實來源 episode 的訓練 reward、validation reward、全期測試報酬、2020／GFC 回撤、全期持股及換手。現存圖仍是歷史輸出，尚未以修正後的實際步數重畫；其中 GFC 的「unseen」標示也不能解讀為 TimeDiff 未看過 2008 年。

## 5. 閱讀結果時的指標與限制

| 項目 | 解讀方式 |
|---|---|
| `valid_final_value` | Validation 最終淨值；各模型起始資金相同，與該期間總報酬排序一致 |
| `valid_sharpe`／`sharpe_0rf` | 日對數報酬平均值除以樣本標準差，再乘 `sqrt(252)`，無風險利率取 0 |
| `mean_stock_weight` | 價格變動後的股票權重平均，即 `1 - cash_weight`；不是單純平均目標 action |
| `mean_daily_turnover` | 目標權重與前一期價格漂移後權重的 L1 距離除以 2，再取日平均 |
| `max_drawdown` | 各統計期間內相對累積高點的最大回撤 |
| `worst_21d_return` | 最差的滾動 21 日累積報酬 |
| `buy_hold_1_11` | 初始現金與 10 檔股票各 1/11，之後買入持有 |
| `buy_hold_1_10` | 初始全股票、每檔 1/10，之後買入持有 |

一般測試是整段連續運行，再切出各年與全期績效，**不是每年重新 reset 或重新訓練**。兩個 buy-and-hold 基準也不是每日恢復等權。

Crash response 診斷固定 `k=10`，每種來源抽取 100 個 episode，比較暴跌前 10 日、後續前 5 日與第 6～21 日持股。它使用訓練期間錨點，屬情境行為診斷；不能代替樣本外測試。且報告的是價格變動後權重，持股下降可能混合「主動減碼」及「股價下跌使股票占比下降」，不能僅憑該數字判定政策主動避險。

2020 年績效不是僅限 COVID 暴跌視窗，2022 年績效也不是單一事件反應。比較時需使用相同期間、相同 checkpoint 選取方式，並保留各 seed 差異；不能將 v1 的 2020–2022 全期直接與延長版的 2020–2025 全期排名混用。

## 6. TRF 與步數修正後的有效範圍

TRF 原先錯把目標現金權重放入分子，應使用交易前、價格漂移後的現金權重。在價格不變、費率 0.25% 時，舊公式使全現金買入股票成本為 0%，全股票清倉成本約 0.500627%；修正後兩者都是 0.25%。這個偏差會影響淨值、選模與有成本訓練的 reward。

| 既有版本／產物 | 是否需要重訓 | 必須更新的結果 |
|---|---|---|
| v1 | **需要**，若要代表正確成本下的訓練 | Validation、選模、2020–2022 測試與診斷 |
| v1_test2025 | 本身不訓練；改用重訓後依 final value 選出的模型 | 2020–2025 測試 |
| v2_sharpe | **需要**，可由同設定的一輪新訓練保存全部 checkpoint，再依 Sharpe 選模 | Validation、選模、測試與診斷 |
| v2_sharpe_test2025 | 本身不訓練；改用重訓後的 Sharpe 所選模型 | 2020–2025 測試 |
| v2_rewardlog | **需要**，舊成本已影響 PPO 更新 | 完整訓練紀錄、validation、checkpoint 評估 |
| v3_nocost | **不因公式修正而需要重訓**，前提是能取回 checkpoint | 全部 300 個 checkpoint 的 validation、重新選模，再測試所選模型及終點；完整 checkpoint 曲線則須全數重測 |
| compare_selection／compare_train_cost | 報表本身不訓練 | 以更新後的來源重建 CSV 與圖 |
| Buy-and-hold 基準 | 不需要 | 使用獨立解析式，此次 TRF 修正不要求重算 |

v1、v2 Sharpe、v2 reward logging 的訓練設定相同時，**只需一輪新的 15-run 有成本訓練，保存全部 checkpoint 及 episode 紀錄**，即可支援不同選模規則與測試終點，不必為每個舊目錄分別重訓。

v3 訓練費率為零，TRF 修正不改變其訓練 reward；訓練採固定步數，也沒有依 validation 提前停止，因此既有模型仍可使用。但驗證費率不為零，舊 `valid.csv` 不能繼續用來選模；只重跑 checkpoint 測試不會自動修復 validation。

對同一個固定 checkpoint，成本共同倍率在權重正規化時抵消，且政策輸入不含淨值，因此持股／換手路徑理論上不因 TRF 修正改變，浮點誤差除外；淨值、報酬、回撤與 Sharpe 必須重算。重新選出的 checkpoint 可能改變，故最終持股比較仍需更新。

步數對齊則是另一件事：舊檔的 `timesteps=500000` 是名義 ID，實際終點為 532,480。修正後依 episode 結束時點分配至實際 checkpoint 區間 `(上一點, 目前點]`，不丟掉尾端 episode；跨界 episode 不拆分。v3 的舊 episode reward 可直接重分桶；v2 的舊 reward 只能用來描述歷史訓練，不能靠重分桶還原正確成本的訓練過程。

完整修正依據與測試記錄見 [TRF 與 reward 步數修正說明](../../examples/README_scenario_mix_corrections.md)。

## 7. 重現入口與本地產物

| 工作 | 正式入口 |
|---|---|
| 一般訓練、最佳模型保存及測試 | `python -m finrl.experiments.train_scenario_mix` |
| 完整 episode 與 checkpoint 紀錄 | `python -m finrl.experiments.train_scenario_mix_logged` |
| 已選定模型延長回測 | `python -m finrl.experiments.evaluate_scenario_mix` |
| 全部 checkpoint 回測 | `python -m finrl.experiments.evaluate_scenario_mix_checkpoints` |
| 彙整與比較 | `analysis.scenario_mix` 下的 `summarize_scenario_mix`、`compare_selection_metrics`、`compare_checkpoint_runs`、`plot_reward_curves`，以 `python -m` 執行 |

完整參數與輸入格式見 [訓練／回測入口](../../finrl/experiments/README_scenario_mix.md) 及 [分析入口](../../analysis/scenario_mix/README.md)。舊 `examples/` 命令保留相容入口。重跑應使用新輸出目錄，以區別修正前後結果。

本地產物狀態（2026-10-12 更新）：

- v1 與 v2 Sharpe 各有 15 個最佳模型；延長回測目錄使用其來源模型。
- `257bf771` 已加入 v2 reward logging／v3 各 300 個 checkpoint，覆蓋所有組別、seed 與 20 個訓練階段；全部 ZIP 完整性與實際步數皆已核對。
- 兩份 TimeDiff 價格 CSV 與 `paths.npz` 已補回原設定路徑，通過價格完整性與合成路徑錨點檢查；後續實驗記錄輸入 SHA-256。
- [第一輪研究 notebook](../../analysis/scenario_mix/scenario_mix_v1_summary.ipynb) 分析 v1 的 **2020–2022**；輸出位置為 `analysis/scenario_mix_v1_export/`。這是分析產物目錄，不是另一個 scenario-mix 訓練版本。

本文件上方各版目錄仍保留修正前的歷史結果。TRF 成對實驗已完成，另存於 `trf_ablation_20261011/`：新舊 TRF 各 15 組從頭訓練，每組實際 532,480 步；連同舊模型，共 1,200 個 checkpoint 均完成兩種成本下的 validation 與測試。15 對訓練的 episode 抽樣完全一致，模型與輸入雜湊等檢查全部通過。

- [實驗解讀與完整圖表](trf_ablation_20261011/report/analysis.md)：固定策略成本效果小，但重訓影響因組別與種子而異。
- [所有選模規則的比較表](trf_ablation_20261011/report/README.md)：final value、Sharpe 與終點模型。
- [完整性檢查](trf_ablation_20261011/audit.json)：模型、實際步數、逐日檔案與成對設定。
