# TimeDiff λ=0.5：步驟 1–3 前測及 D 校準

本輪僅使用 2005–2017 訓練與 2018 驗證；未讀取 2019 合成測試路徑、未執行 2020／2022 真實事件回測。
seed=0 僅作開發前測。本輪使用 CPU、PPO EIIE 特徵、16 個平行環境；每環境 rollout=504、每更新 8,064 次互動。
R／H／C／最後選定的 D 各跑 32 更新=258,048 次互動；D 固定重複候選各跑 8 更新=64,512 次互動。此為完整 21 日 episode 對齊，原暫定正式步數仍待下一階段確認。

## 步驟 1：資料及環境

觀測僅含 50 日 HLC 與前次實際配置；每條訓練 episode 21 步，不添加額外終止 reward。初始全現金，首次買入照常計費。
費用以自融資方程 mu=1-c*sum(abs(mu*target_stock-previous_stock)) 求解；reward=100*log(扣費後資產比)。股票及現金配置非負且合計為一。
RL 使用當前觀測窗最後可知 close 作相對價格標準化；不額外擬合使用未來資料的 scaler。無預測目標、事件標籤或來源旗標進入 observation。

## 步驟 2：R／H／C／D

同一 seed 的初始策略參數雜湊全部一致。每個事件獨立訓練；R/H 在同事件 E0/E1 的訓練資料及初次時程相同，引用同事件的相同 checkpoint，不跨事件移轉。
H/C/D 使用八個真實與八個合成活躍環境，每個 episode 等長，因此已完成的 episode 與互動比例均精確為 50%。C/D 使用相同合成路徑池與總互動數。

## 步驟 3：D 參數及選擇依據

先在 16 條環境路徑池比較固定重複 1/2/4/8 個完整更新週期。每次 PPO 更新都重新收集 rollout，不重用舊 rollout。
校準主指標是相對同步 C 策略，在固定、不重疊的真實 2018 驗證 episode 上更新 5–8 的平均配對改善。合成 2018 驗證、重播池改善與未改善紀錄另外保存。
若重播改善大於其標準誤、但真實驗證退步大於其標準誤，對較佳重複週期再比較 32/64 條路徑池；活躍環境仍為 16，保持 PPO 更新批次固定。
池大小與最短更新數採配對一個標準誤規則偏向較小值；改善及退步門檻取驗證更新配對標準誤的中位數及第 90 百分位。

```json
{
  "environment_pool_size": 16,
  "active_parallel_environments": 16,
  "minimum_repeats": 2,
  "maximum_repeats": 2,
  "improvement_threshold": 0.057570754160597176,
  "regression_threshold": 0.09285618412102453,
  "plateau_patience": 2,
  "threshold_units": "sum of 100*log(net wealth ratios) over a 21-day episode",
  "threshold_estimation": "median / 90th percentile of paired real-2018 validation update standard errors",
  "best_fixed_repeat": 2,
  "best_raw_pool": 16,
  "larger_pool_triggered": true,
  "calibration_interactions_per_candidate": 64512,
  "selection_score": "mean paired real-2018 validation gain over updates 5-8 versus C at the same step",
  "pooled_case_score": 0.1380178444029097,
  "formal_parameters_locked": false,
  "limitation": "single development seed, few non-overlapping real-2018 episodes; no event performance or 2019 evaluation"
}
```

這些是本輪開發校準數值；正式設定尚需依原計畫步驟 4 的耗時、收斂與正式種子規劃鎖定。沒有強制把 D 結果解讀為優於 C。
adaptive D 遇到平台（連續兩次未超過改善門檻）、驗證退化或最長次數即切換路徑池，且僅在完整 episode／PPO 更新邊界切換。原因及所有未改善更新均記錄於 updates.csv。

## 交付

- [D 候選比較](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/D_calibration_candidates.csv)。
- [選定參數](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/D_selected_parameters.json)。
- [所有工作摘要](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/job_summary.csv)。
- [24 組配對 checkpoint 索引](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/paired_group_checkpoints.csv)。
- [資料接入稽核](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/step1_data_adapter.json)。
每個工作保存 initial/best/latest checkpoint、optimizer、每次更新的驗證向量、pool 排程、RNG 與環境續跑狀態。


## 獨立核對與 adaptive D 結果

10 項環境／記帳測試及實際 checkpoint 續訓一致性通過。共核對 54 個已完成工作、6,967,296 次訓練互動；所有工作初始策略參數雜湊一致。
資料檔案、每更新步數、完整 episode、來源 50% 比例、checkpoint 內 optimizer、RNG 與環境存檔均核對。
真實驗證有 9 條目標互不重疊的 2018 episode，評分目標從 2018-03-15 至 2018-12-12。為使歷史與未來完整位於 2018，前 50 日僅作觀測；尾端不足 21 日不組成 episode。這不是全年報酬；仍是小樣本且所有調參共用同一驗證年度，不能當成獨立泛化檢定。

每個完整更新包含每個活躍環境 24 條完整 21 日 episode。16 路徑的固定 D 池中，每條路徑在一次更新內被走 24 次；重複 k 個更新週期代表同一路徑最多 24×k 次 episode 訪問，且每次更新都重新用新策略收集 rollout。這個計數必須與「k 個 PPO 更新」分開解讀。

### 排程選擇的彙總證據

| 固定池16：重複更新數 | 相對 C 的配對分數改善 |
|---:|---:|
| 1 | +0.022326 |
| 2 | +0.138018 |
| 4 | +0.086696 |
| 8 | +0.083945 |

固定較佳重複數 2，再比較池大小：

| 路徑池大小 | 相對 C 的配對分數改善 |
|---:|---:|
| 16 | +0.138018 |
| 32 | +0.039947 |
| 64 | +0.001719 |

以上分數先對同一條真實 2018 驗證 episode 的六個條件取平均，再取驗證 episode 平均；不把六種條件重複視為六倍獨立市場樣本。池大小是固定最佳重複數後的分段搜索，並非完整池大小×重複次數的全因子搜索。

**本輪選出的範圍為 [2, 2]，實際排程是固定每 2 個更新換池。** 平台與驗證退步門檻仍作記錄，但在此退化範圍下不會改變換池時點；本輪沒有證明自適應提前停用比固定週期有額外效益。

| 條件 | C 最終驗證分數 | D 最終驗證分數 | D−C 最終差 | 更新29–32平均差 |
|---|---:|---:|---:|---:|
| crash_2020_E0_f0 | 0.705879 | 0.245377 | -0.460503 | -0.423457 |
| crash_2020_E1_f0 | 0.326609 | 0.419256 | +0.092648 | -0.037312 |
| rebound_2020_E0_f0 | -0.602203 | 0.491559 | +1.093762 | +0.974330 |
| rebound_2020_E1_f0 | 0.443645 | -0.690093 | -1.133738 | -0.842203 |
| bear_2022_E0_f0 | 0.077756 | 0.049740 | -0.028016 | -0.035384 |
| bear_2022_E1_f0 | 0.105045 | -0.394153 | -0.499198 | -0.633616 |

在固定 258,048 次互動的最後策略比較，D 的真實 2018 驗證分數高於 C 的條件為 2/6；其餘未改善結果完整保留，不把選出參數當成 D 已普遍有效。
表中分數單位為 21 日累積的 100×log 扣費後資產比，不是事件報酬率，也不是年化報酬率。

### 依既定 2018 選模規則保存的最佳 checkpoint

| 條件 | C 選定更新 | D 選定更新 | C 最佳驗證 | D 最佳驗證 | D−C |
|---|---:|---:|---:|---:|---:|
| crash_2020_E0_f0 | 20 | 14 | 0.722248 | 0.626295 | -0.095953 |
| crash_2020_E1_f0 | 14 | 18 | 0.788486 | 0.636276 | -0.152210 |
| rebound_2020_E0_f0 | 0 | 31 | 0.344365 | 0.536894 | +0.192528 |
| rebound_2020_E1_f0 | 22 | 21 | 0.530685 | 0.361846 | -0.168840 |
| bear_2022_E0_f0 | 5 | 14 | 0.427812 | 0.700765 | +0.272953 |
| bear_2022_E1_f0 | 11 | 9 | 0.575473 | 0.535958 | -0.039515 |

按各自最佳 2018 checkpoint 比較，D 較 C 高的條件也是 2/6。這是同一驗證資料上的選模分數，不能當作未見測試績效；選模資料未使用真實事件或 2019。
部分曲線長訓練後回落，最佳與最後 checkpoint 都保留；此結果提示在下一階段確認訓練長度與收斂，不能以初期校準改善直接外推至長期訓練。

![2018驗證曲線](/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/real2018_learning_curves.png)
