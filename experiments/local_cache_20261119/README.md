# FinRL 本機訓練快取快照（2026-11-19）

保存同步遠端 `main` 前，本機尚未提交的 93 組訓練快取，共 465 個檔案、80,172,656 bytes。這批檔案在遠端已有同名版本，但內容雜湊不同，因此將本機版本獨立保存。原本 `results/ppo_synthetic_vs_real/train_cache/` 下的同名快取採用遠端版本，兩份結果均保留。

| 內容 | 數量 |
|---|---:|
| `best_model.zip` | 93 |
| `last_model.zip` | 93 |
| `status.json` | 93 |
| `training_log.csv` | 93 |
| `validation_log.csv` | 93 |

`train_cache/` 保留各 run 原有目錄名稱與逐位元內容，包含原本已追蹤但被修改的一份 `best_model.zip`。`status.json` 內的路徑仍指向原始訓練位置；此目錄作為快照，不直接充當目前 runner 的快取池。

`manifest.json` 記錄原始相對路徑、快照路徑、大小、SHA-256、擷取時本機 commit 與對照的遠端 commit。擷取時 commit 表示 checkout 狀態，不宣稱所有模型均由該版本程式訓練。這次操作未重新訓練或重新評定績效。

完整性檢查結果見 `validation.json`；在此目錄執行以下指令可重新核對檔案：

```bash
sha256sum -c SHA256SUMS
```

本次亦提交 Python 環境設定與原 notebook 的 kernel 名稱變更，並保留遠端 notebook 新增的內容及輸出。

既有 `analysis/scenario_mix_v1_summary.ipynb` 是相對符號連結，目標為 `scenario_mix/scenario_mix_v1_summary.ipynb`，擷取時目標不存在。本次保留並追蹤連結本身；沒有可一併提交的目標 notebook，狀態亦記錄於 manifest。
