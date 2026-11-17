# TimeDiff λ=0.5：PPO reward curves（seed=0）

D 的實際訓練 reward 在六個條件均上升：最後 4 次 rollout 的平均皆高於最初 4 次。此判斷是描述性比較，不能取代獨立驗證或證明跨種子改善。

| 情境 | 前4次平均 | 後4次平均 | 改善 | 同一訓練池更新後改善次數 |
|---|---:|---:|---:|---:|
| 2020 crash ／ E0 | -14.754 | -11.798 | +2.956 | 32/32 |
| 2020 crash ／ E1 | -15.598 | -12.640 | +2.958 | 29/32 |
| 2020 rebound ／ E0 | 1.802 | 3.468 | +1.666 | 27/32 |
| 2020 rebound ／ E1 | 1.175 | 2.231 | +1.056 | 31/32 |
| 2022 bear ／ E0 | -9.578 | -8.180 | +1.398 | 27/32 |
| 2022 bear ／ E1 | -9.175 | -8.389 | +0.786 | 29/32 |

固定資料的混合 Train／Val／Test 在六個條件均較初始化改善；真實 2018 Val 僅 2/6、真實 2019 Test 僅 2/6 較初始化改善。主要改善來自合成情境，尚不能據此宣稱真實行情泛化或指定交易行為已改善。

## 圖表與 reward 定義

- `actual_training_reward_R_H_C_D.png` 是原始 SB3 訓練紀錄：含隨機探索、最近 100 個完整 episode 的平均。每列在當次梯度更新之前記錄；CSV 同時提供收集步數及實際使用策略的訓練步數。最後一列由 checkpoint 31 收集，尚未反映第 32 次更新後策略。
- H/C/D 整體訓練 episode 比例是 50% 真實、50% 合成；logger 最後 100 個 episode 的移動視窗未必恰好各半。
- `R/H/C/D_train_val_test.png` 使用固定路徑、確定性策略比較三個 split。R 僅真實資料；H/C/D 各 split 均為 50% 真實平均 reward + 50% 該組合成平均 reward。不是將不同數量的 episode 直接混平均。
- `common_real_R/H/C/D_train_val_test.png` 只用各組完全相同的真實資料，可比較一般行情表現。R/H 的模型在同一情境 E0/E1 共用，因此每情境只畫一次。
- `D_same_pool_update_gains.png` 在每次更新前後用同一組保留的訓練價格路徑評估，排除該次池抽樣差異；D 每兩次更新換池，不將跨池 gain 加總為學習幅度。
- reward 是一個 21 日 episode 的日 reward 總和：`100 × ln(V21 / V0)`，再跨 episode 平均；不是日均 reward、簡單報酬百分比或年化績效。費率 0.0025、初始全現金，初次建倉也扣費。負 reward 向零改善也屬上升。
- 原始 train reward 有探索噪音且池持續變動；其數值不能直接與確定性 val/test 的高低作過擬合判斷。固定 split 評估曲線可減少這個混淆。

## 固定評估資料

- Train：從 3,202 個 2005–2017 真實訓練 episode 等距選 64 個，未來交易日互不重疊；合成訓練從各 800 條中以固定 seed 20261115 選 64 條。是固定樣本評估，不是全訓練集平均。C/D 使用完全相同的真實及合成路徑。抽樣在評估任何 checkpoint 前鎖定。
- Val：原先的 2018 真實 9 個非重疊 episode 及合成 200 條，沿用訓練時的確定性評估向量；所有 18 組初始與最終 checkpoint 重新評估並核對。
- Test：本次依要求新增 2019 真實 9 個非重疊 episode，以及每條件先前已獨立生成的 200 條 test 路徑。真實資料的 50 日歷史與 21 日目標均在 2019 年。
- 2019 真實評分日期：2019-03-15～2019-04-12; 2019-04-15～2019-05-14; 2019-05-15～2019-06-13; 2019-06-14～2019-07-15; 2019-07-16～2019-08-13; 2019-08-14～2019-09-12; 2019-09-13～2019-10-11; 2019-10-14～2019-11-11; 2019-11-12～2019-12-11。這是 episode 評估，不是全年連續持倉回測。
- 合成 test 使用 2019 背景，但 C/D 條件仍包含事件實現值建立的事後情境標籤；合成測試改善不能視為真實市場事件交易改善。合成路徑共用背景時有相關性，不能當成獨立重複實驗。
- 新增 test 曲線僅供診斷，不用來選 checkpoint、重新訓練或調 D。正式 2020/2022 真實事件尚未回測。2019 現已被檢視，後續若據此改動設定需另記為探索分析。

## D：固定資料上的初始至最終改善

| 情境 | 混合Train Δ | 混合Val Δ | 混合Test Δ | 真實Val Δ | 真實Test Δ | 最終真實Test D−C |
|---|---:|---:|---:|---:|---:|---:|
| 2020 crash ／ E0 | +6.948 | +7.052 | +6.654 | -0.099 | -0.843 | -0.317 |
| 2020 crash ／ E1 | +6.461 | +7.358 | +6.705 | +0.075 | -0.846 | -0.310 |
| 2020 rebound ／ E0 | +2.851 | +2.527 | +2.469 | +0.147 | -0.372 | -0.875 |
| 2020 rebound ／ E1 | +3.345 | +3.421 | +4.233 | -1.034 | +1.141 | +0.599 |
| 2022 bear ／ E0 | +0.909 | +1.221 | +1.102 | -0.295 | -0.454 | +0.485 |
| 2022 bear ／ E1 | +0.343 | +0.357 | +0.712 | -0.739 | +0.034 | +0.105 |

## 產出與可追溯性

- `all_reward_curves.pdf`：全部圖的多頁 PDF。
- `checkpoint_reward_curves.csv`：每個條件、組別、checkpoint、split 的真實、合成及固定比例混合平均值。
- `rollout_training_rewards.csv`：原始訓練 reward 與時間對齊資訊；`D_training_trends.csv`：上表數據。
- `initial_vs_final.csv`、`paired_D_vs_C.csv`：初末差值及同 checkpoint D/C 配對比較，沒有依 test 挑選 checkpoint。
- `evaluations/*.npz`：逐 checkpoint、逐 episode reward；`evaluation_manifest.json`：固定選樣、日期、來源 SHA-256 與 checkpoint SHA-256；`audit.json`：完整性與來源未變動核對。
- `source_snapshot.py` 保留最初評估程式；`reporting_source_snapshot.py` 保留修正報告排版與續跑狀態紀錄後的程式。兩者雜湊另記於 audit；評估計算相同。
- 原有模型、訓練紀錄、資料銀行及先前報告均保留原狀。

圖檔：

- [R_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/R_train_val_test.png)
- [H_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/H_train_val_test.png)
- [C_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/C_train_val_test.png)
- [D_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/D_train_val_test.png)
- [common_real_R_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/common_real_R_train_val_test.png)
- [common_real_H_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/common_real_H_train_val_test.png)
- [common_real_C_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/common_real_C_train_val_test.png)
- [common_real_D_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/common_real_D_train_val_test.png)
- [actual_training_reward_R_H_C_D.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/actual_training_reward_R_H_C_D.png)
- [D_same_pool_update_gains.png](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/reward_curves_20261115/D_same_pool_update_gains.png)
