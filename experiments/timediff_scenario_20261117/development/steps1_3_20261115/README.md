# TimeDiff λ=0.5：步驟 1–3 執行入口

工作在 tmux 主機程序執行，可斷線後繼續。

```bash
tmux -S /mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115/tmux.sock attach -t timediff-steps1-3-20261115
```

`progress` 視窗顯示任務階段與工作進度；完成後可查看本目錄 `REPORT.md`。

- `status.json`：目前階段、工作清單及最後狀態。
- `master.log`：總日誌；`jobs/<工作名>/training.log`：單項完整訓練日誌。
- `step1_data_adapter.json`、`banks/manifest.json`：接入與資料來源稽核。
- `environment_tests.json`：環境／財務記帳測試。
- `integration_smoke/audit.json`：真正 PPO 更新後的斷點恢復一致性檢查。
- `D_selected_parameters.json`：D 校準結束後產生的選定數值與選擇規則。
- `D_calibration_candidates.csv`：所有候選的配對驗證比較。
- `paired_group_checkpoints.csv`：完成後的 R/H/C/D checkpoint 索引。

本輪僅用 2005–2017 真實與合成訓練資料、2018 真實及合成驗證資料。未讀取 2019 合成測試路徑，未執行 2020/2022 真實事件回測。

初次前測每模型 32 個 PPO 更新、258,048 次互動；D 校準候選各 8 個更新、64,512 次互動。各完整 rollout=16×504；每條 episode 21 日。PPO 的 reward 僅為扣費後 100×log 資產比，費率0.0025，初始建倉計費。

主機重啟或程序失敗時可重啟同一命令；完成工作會略過，未完成工作從最近的完整更新 checkpoint（含 optimizer／RNG／環境）續跑：

```bash
/opt/anaconda3/envs/finrl_exp/bin/python -u -m finrl.experiments.timediff_scenario_pilot run --out /mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115 --workers 2 --resume
```

在 FinRL_Exp 專案根目錄執行恢復命令。執行中的來源碼雜湊已固定；程式設定變更時會拒絕沿用不相容的工作快取。
