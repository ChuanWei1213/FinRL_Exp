# TimeDiff λ=0.5 × PPO：三個E0、10 seeds

使用既有λ=0.5合成資料，只保留急跌、反彈、熊市的E0。PPO seeds1–10；seed0僅作D校準及開發，不納入10種子結果。所有模型重新隨機初始化，各情境獨立訓練。

## 資料與設定

- Train：沿用上一輪完全相同的2005–2017價格路徑銀行；真實3181條，各E0合成798條，H歷史合成800條。少數原始生成路徑因無法在訓練分割內提供71日歷史，沿用既有排除紀錄。
- Val：評分目標日期2018-01-01～2019-12-31，實際交易日2018-01-02～2019-12-31。24個完整21日窗口覆蓋503個交易日；最後一窗為覆蓋尾端而與前窗重疊1日。觀測與前置交易只使用各目標當時已知的過去資料，可來自2017年。
- 合成Val：每條件合併2018原validation200條、2019原test200條，共400條；2019現在是validation，這輪沒有2019獨立test的宣稱。
- 每個episode：50日觀測 → k日真實交易 →21日真實／合成交易；k每次reset均勻抽整數0–21。H/C/D一半來源episode為合成，前置日也是真實價格，所以實際合成交易步數約三分之一。
- 16個環境、rollout每環境504步、minibatch504、每更新10 epochs；每個主要模型32更新＝258048步。reward=100×log扣費後淨值比，費率0.0025，初始全現金，初次建倉扣費。預測、來源旗標、k不進入觀測。
- 同seed各組初始權重相同，C/D使用相同合成銀行、k序列及互動預算。不同seed的初始權重、路徑抽樣、k與D池RNG均不同。
- 批次環境已逐步核對原版費用、持倉、觀測、終止與reset；D換池時讓未完成episode自然結束。續訓恢復策略、optimizer、環境與RNG的檢查另存。
- 真實Val對每個目標平均全部22個k，再對原目標取平均作選模。合成Val與固定train樣本用事先固定、平衡的k。學習曲線的Train是固定64條真實／64條合成路徑的評估，並非整個訓練庫平均。

## D校準與模型選擇

- 先用seed0及新Val比較重複1/2/4/8次，必要時測較大池。鎖定 pool=16、最短2／最長4次更新；改善門檻0.040891、退步門檻0.059320。全部主實驗使用同一設定，不逐seed按事件績效調參。
- 主結果使用2018–2019真實Val平均suffix reward最高的checkpoint，同分取較早，允許初始化被選中；final32為事先指定的敏感度分析。120個模型的兩套名單在事件策略損益計算前鎖定。
- 三個E0條件仍使用各事件首21日實現報酬／年化波動率，不是整段事件條件。Test固定同一首21日的真實行情，由全現金開始、確定性動作、不插入額外交易日、不做分段預測或續訓。初次建倉成本正常計入。

| 情境 | 真實Test區間 | 條件市場報酬(未扣費) | 條件年化波動率 |
|---|---|---:|---:|
| 2020 crash | 2020-02-13～2020-03-13 | -17.24% | 68.12% |
| 2020 rebound | 2020-03-17～2020-04-15 | +23.00% | 58.53% |
| 2022 bear | 2021-12-30～2022-01-28 | -6.11% | 14.64% |

## 主結果：驗證選模

扣費後簡單報酬，平均±seed標準差；每組每情境n=10。全現金=0%。

| 情境 | R | H | C | D | 每日等權 |
|---|---:|---:|---:|---:|---:|
| 2020 crash | -16.34±2.61% | -17.64±2.91% | -15.63±0.62% | -15.87±0.37% | -17.50% |
| 2020 rebound | +20.33±1.29% | +20.47±1.39% | +21.35±2.42% | +21.27±1.83% | +22.59% |
| 2022 bear | -6.68±0.99% | -5.87±0.78% | -5.86±0.09% | -6.00±0.61% | -6.39% |

配對差異按同一seed相減；區間為95% t區間(df=9)，單位為百分點。

| 情境 | 比較 | 平均差 | 95%區間 | 勝／平／負 |
|---|---|---:|---|---:|
| 2020 crash | C-R | +0.71 | [-1.05, +2.48] | 6／0／4 |
| 2020 crash | D-R | +0.47 | [-1.38, +2.32] | 5／1／4 |
| 2020 crash | D-H | +1.77 | [-0.35, +3.89] | 7／0／3 |
| 2020 crash | D-C | -0.25 | [-0.66, +0.17] | 3／2／5 |
| 2020 rebound | C-R | +1.02 | [-0.99, +3.03] | 6／0／4 |
| 2020 rebound | D-R | +0.94 | [-0.35, +2.24] | 8／0／2 |
| 2020 rebound | D-H | +0.80 | [-0.61, +2.22] | 7／0／3 |
| 2020 rebound | D-C | -0.08 | [-1.92, +1.77] | 4／0／6 |
| 2022 bear | C-R | +0.81 | [+0.11, +1.52] | 8／1／1 |
| 2022 bear | D-R | +0.67 | [-0.14, +1.49] | 7／1／2 |
| 2022 bear | D-H | -0.13 | [-0.84, +0.57] | 3／1／6 |
| 2022 bear | D-C | -0.14 | [-0.59, +0.31] | 2／4／4 |

## 敏感度：最後32次更新

扣費後簡單報酬，平均±seed標準差；每組每情境n=10。全現金=0%。

| 情境 | R | H | C | D | 每日等權 |
|---|---:|---:|---:|---:|---:|
| 2020 crash | -13.81±5.40% | -15.45±5.79% | -7.03±3.80% | -6.13±2.66% | -17.50% |
| 2020 rebound | +16.44±5.75% | +16.05±5.15% | +22.17±2.65% | +20.62±2.86% | +22.59% |
| 2022 bear | -5.36±1.83% | -4.75±1.00% | -2.71±1.50% | -2.97±2.40% | -6.39% |

配對差異按同一seed相減；區間為95% t區間(df=9)，單位為百分點。

| 情境 | 比較 | 平均差 | 95%區間 | 勝／平／負 |
|---|---|---:|---|---:|
| 2020 crash | C-R | +6.78 | [+2.81, +10.76] | 9／0／1 |
| 2020 crash | D-R | +7.68 | [+3.27, +12.09] | 9／0／1 |
| 2020 crash | D-H | +9.32 | [+4.55, +14.08] | 9／0／1 |
| 2020 crash | D-C | +0.90 | [-2.38, +4.18] | 5／0／5 |
| 2020 rebound | C-R | +5.73 | [+1.09, +10.36] | 9／0／1 |
| 2020 rebound | D-R | +4.17 | [+0.74, +7.60] | 9／0／1 |
| 2020 rebound | D-H | +4.56 | [+0.32, +8.80] | 7／0／3 |
| 2020 rebound | D-C | -1.55 | [-4.39, +1.28] | 3／0／7 |
| 2022 bear | C-R | +2.64 | [+1.58, +3.71] | 10／0／0 |
| 2022 bear | D-R | +2.38 | [+0.49, +4.28] | 9／0／1 |
| 2022 bear | D-H | +1.78 | [+0.19, +3.37] | 8／0／2 |
| 2022 bear | D-C | -0.26 | [-2.22, +1.70] | 5／0／5 |

## 主結果：風險、成本及初始化選中數

| 情境／組別 | 平均最大回撤 | 平均成本 | 平均股票曝險 | 選到初始化 |
|---|---:|---:|---:|---:|
| 2020 crash / R | -22.52% | 0.357% | 87.63% | 3/10 |
| 2020 crash / H | -24.31% | 0.390% | 93.20% | 2/10 |
| 2020 crash / C | -22.40% | 0.295% | 89.61% | 4/10 |
| 2020 crash / D | -22.72% | 0.291% | 90.30% | 4/10 |
| 2020 rebound / R | -5.52% | 0.377% | 89.31% | 2/10 |
| 2020 rebound / H | -5.49% | 0.374% | 89.54% | 4/10 |
| 2020 rebound / C | -7.36% | 0.467% | 95.00% | 0/10 |
| 2020 rebound / D | -7.25% | 0.463% | 95.36% | 0/10 |
| 2022 bear / R | -8.89% | 0.362% | 83.30% | 1/10 |
| 2022 bear / H | -8.44% | 0.307% | 89.38% | 4/10 |
| 2022 bear / C | -7.94% | 0.275% | 90.82% | 8/10 |
| 2022 bear / D | -8.17% | 0.300% | 90.21% | 4/10 |

## 解讀限制與行為

- 95%區間只描述同一組生成資料、同一真實事件下PPO隨機種子的差異，不涵蓋生成器、預測或市場抽樣不確定性；10個seed不是10次獨立市場事件，三個情境也不合併成30個獨立市場樣本。
- 條件由事後實現值建立，屬情境機制實驗；seed0的真實事件在先前開發已被查看，這輪不宣稱是從未看過事件的前瞻檢驗。這次沒有依事件結果調參、重抽條件或選checkpoint。
- test_behavior.csv只用決策收盤日已知的市場訊號，統計其後交易的曝險調整，排除價格自然漂移。超過0.5個百分點才算有效。同一事件訊號在10個seed重複出現，不能視為10倍獨立行為樣本；固定曝險再平衡也可能產生同向交易，因此次數本身不能證明已學會指定交易行為。

## 檔案與驗證

主實驗120模型、合計30,965,760步。從tmux啟動至報告約53.5分鐘，不含事前程式修改、測試與CPU計時。
- test_summary.csv、paired_comparisons.csv：平均、標準差、配對95%區間與勝平負。
- test_per_seed.csv、test_daily_trades.csv、test_behavior.csv：逐seed績效、交易、曝險與行為。
- learning_curves.csv：全部主要模型的固定train/val評估；selected_checkpoints.csv、test_selection_locked.json：事先鎖定的模型及SHA-256。
- main_job_summary.csv、checkpoint_inventory.json：訓練工作與checkpoint清單；D_parameters_locked.json及D_calibration_candidates.csv：seed0校準。
- audit.json、integration_smoke/audit.json、environment_tests.log：資料與結果完整性、種子配對及續訓核對。
- charts.pdf：學習曲線、真實Test淨值與曝險；陰影是seed標準差，並非市場信賴帶。

- [R_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/R_learning_curves.png)
- [real_only_R_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/real_only_R_learning_curves.png)
- [H_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/H_learning_curves.png)
- [real_only_H_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/real_only_H_learning_curves.png)
- [C_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/C_learning_curves.png)
- [real_only_C_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/real_only_C_learning_curves.png)
- [D_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/D_learning_curves.png)
- [real_only_D_learning_curves.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/real_only_D_learning_curves.png)
- [best_validation_test_wealth.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/best_validation_test_wealth.png)
- [best_validation_test_exposure.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/best_validation_test_exposure.png)
- [final32_test_wealth.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/final32_test_wealth.png)
- [final32_test_exposure.png](/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116/final32_test_exposure.png)
