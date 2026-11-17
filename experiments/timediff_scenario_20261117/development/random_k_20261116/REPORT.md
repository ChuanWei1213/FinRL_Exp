# 隨機前置 k=0–21 日：PPO 重訓與 D 重新校準

本輪使用既有 TimeDiff λ=0.5 合成路徑，從隨機初始化重新訓練 PPO，僅 seed=0 開發前測。舊 k=0 結果保留原狀；沒有執行正式 2020/2022 真實事件回測。

## 設定與公平比較

- 每次 reset 抽整數 k∈{0,…,21}。50 日觀測歷史 → k 日真實交易 → 完整 21 日真實／合成交易。前置日正常計費與計算 reward；進入合成段不重設資金、持倉或觀測窗口。
- PPO 仍使用相同 EIIE、50 日 HLC 與前次配置、初始全現金、費率 0.0025、每日 100×log 扣費後資產比。預測／來源旗標／k 不進入 observation。
- 每更新 16×504=8,064 次實際互動；主要模型各 32 更新=258,048 次。跨 rollout 的未完成 episode 自然延續。相同 seed 起始權重一致、C/D 相同互動預算與價格路徑庫。
- 真實／合成配對環境共用 k 抽樣序列，以維持精確 50% 的來源 episode 與來源環境互動數；不同配對使用不同 RNG。k 與路徑抽樣 RNG 分開。因合成來源 episode 的前置 k 日也是真實行情，實際合成交易日比例約為總互動的三分之一，逐工作另報實際數量。
- D 每次更新重新收集 rollout。達換池條件後，已開始的 episode 沿原路徑走完，最多 41 個交易步後使用新池；不把未完成交易截斷。實際延續舊池的步數與換池理由逐更新保存。
- 使用 18/24 邏輯 CPU 的 affinity 上限、8 個程序×每程序 2 個 torch 執行緒、較低排程優先度；現有 CUDA13 與驅動不相容，未使用 GPU，也未改動驅動或系統套件。

## 資料、驗證及指標

- 50+k 日所需的額外歷史由原始真實行情補齊。先要求所有 k 都能合法使用該路徑，再抽 k；沒有為了可用路徑而重抽較小 k。
- 真實訓練 3181 條，真實 2018 驗證 8 個非重疊目標；真實 2019 診斷測試 8 個。歷史與目標均位於各自資料分割內。
- 既有合成保留路徑中，少數位於分割起點、無法提供 71 日歷史者排除；原始 ID 與排除數均在銀行 manifest。未重新生成或更換預測目標，品質門檻不變。
- 全 episode reward 含 k+21 日，另外保存最後 21 日 suffix reward。suffix 從前置日結束時的實際資金／持倉續算，包含當日交易成本；不是在合成起點重新建倉。
- 2018 選模與 D 校準主指標為最後 21 日的純 reward。真實驗證對每個目標平均全部 22 個 k，再以原目標為樣本單位計算標準誤，不把 k 視為獨立行情樣本。合成驗證／測試各路徑使用固定、平衡分配的 k；抽樣不依績效。
- train 曲線為固定 64 條真實／64 條合成路徑的確定性評估；val/test 為各合法保留路徑。H/C/D 的混合曲線為兩來源平均 reward 各佔 50%。另提供各組相同真實資料的曲線，以及原始隨機探索訓練曲線。
- 2019 只在 D 設定決定後進行診斷評估，不參與選模或調參。2019 先前已被檢視，不能再宣稱為全新未見測試。所有情境標籤仍屬事後機制實驗。

## D 重新校準

重複更新數比較 1/2/4/8；依 2018 配對改善決定是否測 32/64 池。此次選擇 pool=16、最短 1、最長 8 次更新。
改善門檻 0.063555，退步門檻 0.094480。仍為單種子前測結果；所有未改善與較差候選保留。

## D 訓練與最終真實資料表現

| 條件 | 原始全episode train：前4→後4 | 原始suffix train：前4→後4 | 真實Val Δ初始 | 真實Test Δ初始 | 最終Val D−C | 最終Test D−C |
|---|---:|---:|---:|---:|---:|---:|
| 2020 crash / E0 | -17.084→-15.951 | -13.978→-13.285 | +0.021 | -0.132 | +0.282 | -0.037 |
| 2020 crash / E1 | -18.600→-15.687 | -15.526→-13.141 | -0.951 | -0.807 | -0.471 | -0.281 |
| 2020 rebound / E0 | -2.149→-0.137 | 1.419→3.065 | -0.473 | +1.672 | -0.651 | +2.583 |
| 2020 rebound / E1 | -2.614→-1.751 | 0.540→2.045 | -0.232 | +0.105 | +0.050 | +0.711 |
| 2022 bear / E0 | -12.146→-11.671 | -9.063→-8.776 | -0.261 | -0.235 | -0.318 | -0.638 |
| 2022 bear / E1 | -12.085→-12.381 | -9.228→-9.495 | +0.191 | -0.336 | +0.234 | -0.383 |

## 與舊 k=0 模型對照及限制

`legacy_k0_common_evaluation.csv` 將兩輪最終模型放在本輪相同的真實目標、全部 22 個 k 上評估，避免直接比較 21 日和 k+21 日總 reward。舊模型的訓練可用路徑與 D 校準不同，因此這不是只改 k 的完整因果消融；結果仍是單種子描述性比較。
隨機 k 排除固定起點的一項捷徑，並不保證生成多樣性、真實泛化或指定增減碼行為。基準現金與每日等權的扣費後分數另存 baselines.csv。

## 產出

完成 54 個工作，共 6,967,296 次訓練互動。自 tmux 啟動至本報告產生約 12.9 分鐘（不含環境修改與事前測試）。
- `reward_curves.csv`、`rollout_rewards.csv`：所有曲線數值與 k、交易日來源、池延續紀錄。
- `D_calibration_candidates.csv`、`D_selected_parameters.json`：本輪 D 校準。
- `job_summary.csv`、`diagnostics/*.json`、`checkpoint_inventory.json`：逐工作／逐 checkpoint 結果與雜湊。
- `environment_tests.log`、`integration_smoke/audit.json`、`audit.json`：環境測試、跨 episode 續跑與結果完整性核對。
- `all_reward_curves.pdf`：完整圖表。

- [R_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/R_train_val_test.png)
- [H_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/H_train_val_test.png)
- [C_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/C_train_val_test.png)
- [D_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/D_train_val_test.png)
- [common_real_R_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/common_real_R_train_val_test.png)
- [common_real_H_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/common_real_H_train_val_test.png)
- [common_real_C_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/common_real_C_train_val_test.png)
- [common_real_D_train_val_test.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/common_real_D_train_val_test.png)
- [actual_training_full_reward.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/actual_training_full_reward.png)
- [actual_training_suffix_reward.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/actual_training_suffix_reward.png)
- [D_same_pool_update_gains.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/D_same_pool_update_gains.png)
- [D_legacy_k0_comparison.png](/mnt/sda/cwei1213/timediff_finrl/random_k_20261116/D_legacy_k0_comparison.png)
