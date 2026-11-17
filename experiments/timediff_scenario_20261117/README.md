# TimeDiff × PPO 情境實驗紀錄（2026-11-17 提交）

本目錄追蹤三個 E0、PPO seeds 1–10 的正式比較，以及先前 seed=0 開發階段；保存實際程式、設定、選模清單、彙總資料、報告與圖表。大型 PPO checkpoint／價格陣列保留在原始本機位置，以 SHA-256 manifest 追蹤。

## 實驗版本

| 版本 | 種子／情境 | k | 驗證／測試用途 |
|---|---|---|---|
| `development/steps1_3_20261115` | seed=0、E0/E1 | 0 | 最初流程及 D 校準；2018 val、2019 診斷 |
| `development/random_k_20261116` | seed=0、E0/E1 | 每 episode 隨機 0–21 | 重訓及重新校準 D；2018 val、2019 診斷 |
| `e0_10seeds_val2018_2019_20261116` | seeds 1–10、三個 E0 | 每 episode 隨機 0–21 | 2018–2019 val；三個真實事件首 21 日 test |

主研究使用凍結的 TimeDiff λ=0.5 資料。三事件各自獨立訓練 R／H／C／D，共 120 個 PPO；每模型 16 env × 504 步 × 32 更新＝258,048 次互動。同 seed 初始權重相同；C／D 共用生成銀行、k 排程與互動預算。H／C／D 有 8 真實＋8 合成環境；觀測為 50 日 HLC 與前次配置，預測不進入觀測。

費率為 0.0025，reward 為 `100 × log(扣費後淨值比)`，初次建倉計費。主選模依 2018–2019 真實 val 最高 reward，同分取較早並允許初始化；final32 為事先指定敏感度。2019 在主研究已屬 val，獨立 test 僅為以下首 21 日：

| 情境 | 真實 test |
|---|---|
| 急跌 | 2020-02-13～2020-03-13 |
| 反彈 | 2020-03-17～2020-04-15 |
| 熊市 | 2021-12-30～2022-01-28 |

E0 條件取自這些窗口的實現報酬／波動，屬事後情境機制實驗。10 seeds 共用同一生成銀行與市場事件，不能視為 10 次獨立市場重複。

## 報告與資料

- [Notion 報告快照](e0_10seeds_val2018_2019_20261116/notion_report_20261116/REPORT.md)：保留擷取時頁面文字與收合結構，圖片及 CSV 改為本目錄相對連結。
- [實際訓練來源拆分補充](e0_10seeds_val2018_2019_20261116/notion_report_20261116/SOURCE_REWARD_SUPPLEMENTS.md)：合成／真實來源兩張圖與逐更新資料。
- `NOTION_SNAPSHOT.md` 保存原始 Notion 格式；`REPORT.local_draft.md` 保存本機生成稿。頁面後續可能有人為編輯，兩者各自保留來源，不以生成稿覆寫快照。
- 主研究根目錄的 `test_per_seed.csv`、`test_daily_trades.csv`、`test_summary.csv`、`paired_comparisons.csv` 保存逐模型、逐日及配對結果。
- `plan.json`、`D_parameters_locked.json`、`D_calibration_candidates.csv`、`test_selection_locked.json`、`selected_checkpoints.csv` 保存預算、校準與選模證據。
- `banks/manifest.json`、`checkpoint_inventory.json` 保存外部價格銀行及 PPO 模型 SHA-256；`external_artifacts.json` 保存原始路徑與版本化檔案對照。
- `validation/pytest.txt` 是本次提交前的相關測試結果；原研究稽核與續訓驗證亦保留於研究目錄。

## 曲線口徑

| 圖表／欄位 | 動作與資料 | reward 期間 |
|---|---|---|
| `charts.pdf`、`learning_curves.csv` | 固定樣本的確定性 train／val 評估；H/C/D 混合各來源 50% | 最後 21 日 suffix |
| 圖 2、`actual_training_curves.csv` | 實際隨機訓練 rollout，最近 100 個完成 episode | 完整 k＋21 日 |
| 圖 2 來源補充、`actual_training_rewards_by_source.csv` | 同一 100 episode 視窗依後 21 日來源拆分；R 無合成來源 | 完整 k＋21 日；另存 prefix／suffix |
| 圖 5 與曝險補充 | 已鎖定 checkpoint 在真實事件的確定性連續交易 | 真實事件首 21 日 |

實際 rollout 的第 u 列由更新 u−1 的策略收集；固定評估使用更新 u 的 checkpoint。來源拆分已核對 3,840 個 checkpoint、384,000 個 buffer entry、每個 k／長度、累計來源與 k 次數、最終環境及 RNG；按每點來源樣本數加權可重組原圖 2。H/C/D 每來源每點為 48–52 個 episode，R 真實為 100 個。獨立 y 軸僅改變圖形尺度。

## 重現

主流程入口：

```bash
/opt/anaconda3/envs/finrl_exp/bin/python -m finrl.experiments.timediff_e0_multiseed \
  run --out /mnt/sda/cwei1213/timediff_finrl/new_e0_run --workers 8 --threads 2
```

完整訓練需依 `external_artifacts.json` 還原 TimeDiff 生成路徑、先前 random-k 銀行及程式內固定的來源根目錄。原始實驗目錄保留在 `/mnt/sda/cwei1213/timediff_finrl`。

本目錄已包含重繪資料。於主研究的 `notion_report_20261116` 子目錄可執行：

```bash
/opt/anaconda3/envs/finrl_exp/bin/python plot_actual_training_reward.py
/opt/anaconda3/envs/finrl_exp/bin/python plot_real_test_exposure.py
/opt/anaconda3/envs/finrl_exp/bin/python plot_training_rewards_by_source.py
```

`extract_training_rewards_by_source.py` 重新核對原始 checkpoint buffer，需要外部模型；繪圖本身使用已納入的來源拆分 CSV。`build_report.py` 是生成稿的來源，Notion 手動編輯另以快照保存。

TimeDiff 生成器、四版本小型模型與生成品質紀錄位於另一個專案：`/home/cwei1213/TimeDiff/experiments/scenario_pilot_20261117`。本機的 λ=1.0 生成比較未重訓或替換本研究的 PPO。

對應 TimeDiff Git commit：`37eadd6c961fcb37c6371b7adc1a9ba635eabcde`。
