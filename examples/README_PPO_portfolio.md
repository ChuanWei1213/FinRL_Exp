# Portfolio Optimization with PPO — 講解與交接文件

用 stable-baselines3 的 PPO 跑 FinRL 的投組最佳化環境，附過擬合偵測與多路徑訓練接口。

---

## 一、要傳的檔案

### 必要

| 檔案 | 說明 |
|---|---|
| `finrl/meta/env_portfolio_optimization/env_portfolio_optimization_gymnasium.py` | gymnasium 相容環境（原環境未修改） |
| `finrl/meta/env_portfolio_optimization/env_portfolio_multipath.py` | 多路徑訓練 wrapper |
| `examples/FinRL_PortfolioOptimization_PPO_Demo.ipynb` | 主 notebook |
| 本檔 `README_PPO_portfolio.md` | |


### 選附

| 檔案 | 說明 |
|---|---|
| `examples/FinRL_PortfolioOptimization_PPO_Demo_fee.ipynb` | 有手續費版本（config B 的執行結果） |
| `examples/make_synthetic_portfolio_data.py` | 合成資料格式範例與驗證函式 |
| `examples/train_portfolio_ppo.py` | 純腳本版（無監控，較簡單） |

### 環境需求

```
python 3.10, stable-baselines3 2.9, gymnasium 1.3, torch, pandas, scikit-learn, quantstats
```

對方需要先 `pip install -e .` 安裝 FinRL。

---

## 二、講解順序（建議 30–40 分鐘）

### 1. 問題設定（5 分）

- **State**：`(3, 10, 50)` — 3 個特徵 × 10 檔股票 × 過去 50 天
- **Action**：`(11,)` — 10 檔權重 + 1 個現金，softmax 後總和 = 1
- **Reward**：`log(V_t / V_{t-1})` — 對數報酬

**為什麼用對數報酬**（這是理解全案的關鍵）：

1. **可加** — `Σ reward = log(V_final / V_initial)`，最大化累積 reward 精確等於最大化最終財富
2. **尺度無關** — 10 萬和 1000 萬賺 1% 的分數相同
3. **自帶風險趨避** — log 是凹函數，`log(0.5) + log(2.0) = 0`，對應「虧 50% 要漲 100% 才回本」

環境裡**沒有任何顯式的風險項**（沒有 Sharpe、沒有波動懲罰），全部的風險控制都來自 log 的凹性。這解釋了後面的實驗結果。

### 2. 資料流程（8 分）

```
YahooDownloader          下載，內部已做除權息還原（OHLC 四欄同係數）
      ↓
存檔凍結                  避免除息追溯改寫
      ↓
OHLC 自洽檢查             low ≤ close ≤ high
      ↓
先切分，再正規化           ← 順序很重要，見下
      ↓
PortfolioOptimizationEnv  算 price_variation = close_t / close_{t-1}
      ↓
(3, 10, 50) 張量
```

**兩個容易踩的坑，值得特別講：**

**坑 1 — 除權息調整必須四個欄位一起做。** 原始 demo notebook 的存檔輸出只調整了 `close`，`open/high/low` 維持未調整，導致 `close < low` 這種不可能的資料。三個特徵 channel 落在不同的價格基準上，餵給網路的是自相矛盾的訊號。notebook 裡的 assert 就是擋這個。

（判斷方式：違規比例。~0.05% 是 Yahoo 的零星壞 tick，clip 掉即可；接近 100% 就是系統性錯誤，資料不能用。）

**坑 2 — 正規化必須先切分再 fit。** `MaxAbsScaler` 除以整段期間的最大值，如果先 fit 再切，訓練時看到的 `close = 0.6` 就隱含「距離**未來**高點還有 40%」。這 10 檔股票有 7 檔的高點落在 2019 之後。

```python
df_train = scaler.fit_transform(raw_train)[COLS]   # 只 fit 訓練期
df_valid = scaler.transform(raw_valid)[COLS]       # 其餘只 transform
```

測試期出現 >1.0 的值是正常的，代表漲破訓練期高點。

### 2.5 環境內部：一個 step 到底發生什麼（8 分）

**繼承關係 —— 原環境完全沒有修改：**

```
PortfolioOptimizationGymnasiumEnv        ← 我們寫的，只覆寫 5 個方法
  └─ PortfolioOptimizationEnv            ← 原 FinRL 環境，一行未動
       └─ gym.core.Env
            └─ gymnasium.core.Env        ← 多重繼承，讓 SB3 認得

MultiPathEnv                             ← 獨立 wrapper，不繼承
  └─ gymnasium.core.Env                     持有多個上述環境，每 episode 抽一個
```

子類別覆寫的只有 `__init__` / `reset` / `step` / `render` / `max_achievable_weight`。
**交易撮合、手續費、reward 計算全部走原環境的程式碼**（`super().step()`），
所以跟原 demo 的 `PolicyGradient` 用的是完全相同的模擬邏輯，兩者可以公平比較。

#### 一個 step 的完整順序

```python
# env_portfolio_optimization.py step()

# 1. 權重正規化
if sum(actions) ≈ 1 and min(actions) >= 0:
    weights = actions                      # 已是合法權重，直接用
else:
    weights = softmax(actions)             # SB3 一定走這條

# 2. 取出「上一步結束時」的實際權重（價格漂移後的）
last_weights = self._final_weights[-1]

# 3. 時間前進一格，載入新的 state 和 price_variation
self._time_index += 1
price_variation = close[t+1] / close[t]    # 現金那格 = 1

# 4. 扣手續費（trf 模型）
mu = solve_fixed_point(...)                # 見下
self._portfolio_value *= mu

# 5. 市場變動改變投組
portfolio = self._portfolio_value * (weights * price_variation)
self._portfolio_value = sum(portfolio)
weights = portfolio / self._portfolio_value    # 漂移後的新權重 → 存入 _final_weights

# 6. 算 reward
rate_of_return = final[-1] / final[-2]     # 同時涵蓋手續費和價格變動
reward = log(rate_of_return) * reward_scaling
```

**時序上沒有未來函數**：agent 在 `t` 時刻看到的 state 只含 `[t-49, t]` 的資料，
做完決策後時間才前進到 `t+1`，賺取 `t → t+1` 的報酬。決策當下不知道 `price_variation`。

#### 手續費：trf（transaction remainder factor）

`mu` 是「交易後剩下多少比例的資產」。它必須用**不動點迭代**解，因為手續費本身
會改變投組價值，而投組價值又決定要交易多少：

```
mu = [ 1 - c·w_0 - (2c - c²)·Σ max(w_prev,i - mu·w_i, 0) ] / (1 - c·w_0)
                                            ↑
                        mu 出現在自己的定義裡 → 迭代到收斂（1e-10）
```

- `c` = `comission_fee_pct`（0.0025）
- `w_0` = 現金權重
- `max(w_prev,i - mu·w_i, 0)` = 只對**賣出**的部分計費

實務含義：**手續費按換手率收，不是按持倉收。** 完全不換倉時 `mu = 1`，
不扣任何費用；這是 Buy & Hold 成本低的原因。

另一個模型 `wvm`（weights vector modifier）較簡化，但**訓練迴圈不能用** ——
原 `PolicyGradient` 讀 `info["trf_mu"]`，只有 `trf` 會產生這個欄位。

#### 兩個容易誤解的點

**① `_actions_memory` vs `_final_weights` 不同**

| | 內容 |
|---|---|
| `_actions_memory[t]` | agent **選擇**的權重 |
| `_final_weights[t]` | 價格漂移**之後**的實際權重 |

手續費是拿 `_final_weights[t-1]` 跟 `_actions_memory[t]` 比對算的 —— 也就是
「從漂移後的位置，移動到新目標位置」需要多少交易。

`return_last_action=True` 餵給 agent 的是 `_actions_memory[-1]`（選擇值），
不是漂移後的實際持倉，所以是個近似。日內漂移約 ±2%，影響不大，
而且原 repo 的 PVM 也是同樣的近似。

**② reward 已經含手續費**

`rate_of_return = final[-1] / final[-2]` 跨越了「扣費」和「價格變動」兩個階段，
所以成本已經反映在 reward 裡。PPO 直接吃這個 reward，**手續費對 PPO 的梯度有貢獻**。

（對比：原 `PolicyGradient` 把 `trf_mu` 當常數存進 buffer，
`log(mu · Σw·y) = log(mu) + log(Σw·y)`，常數項微分為 0 ——
**手續費對它的學習訊號零貢獻**，調 `comission_fee_pct` 不會讓它學會少換倉。）

### 3. 為什麼需要一個 wrapper（5 分）

`PortfolioOptimizationGymnasiumEnv` 做四件事：

| 問題 | 解法 |
|---|---|
| 原環境用舊 `gym`，SB3 2.x 只認 `gymnasium` | 多重繼承 + space 轉換 |
| `Box(0,1)` 下 softmax 動態範圍只有 `e^1`，單一資產權重卡在 **0.21** | `action_space_mode="symmetric"` → `Box(-1,1)` 再乘 `action_scale=5` |
| 每個 episode 結束畫圖 + quantstats，訓練期間觸發數百次 | `plot_on_terminal=False` |
| — | `max_achievable_weight()` 診斷用 |

第二點值得展開：11 個資產時 `e/(e+10) = 0.21`，agent **永遠無法集中持倉，也無法全部持有現金**。這是 SB3 的 action clipping 和環境 softmax 交互作用的結果，不看數學算不出來。

### 4. 兩個非顯而易見的修正（8 分）

這兩個都是實驗跑砸之後診斷出來的。

**修正 A：agent 原本看不到自己的持倉**

`return_last_action` 預設 `False`，觀測只有價格張量。但環境**按換手率收手續費** — 等於蒙著眼睛被罰款，agent 無從判斷該換多少倉。

```python
return_last_action=True     # 觀測變成 Dict
POLICY = "MultiInputPolicy" # Dict 觀測必須用這個
```

EIIE 透過 Portfolio Vector Memory 明確拿到這個資訊，PPO 版本漏掉了。

**修正 B：噪音蓋過訊號**

`action_scale=5` 同時放大平均值**和**噪音。SB3 預設 `std=1`：

```
logit 空間噪音 = std × action_scale = 1 × 5 = 5.0
logit 空間範圍 =                        [-5, +5]
```

噪音跟整個有效範圍一樣大。實測後果：

```
訓練 rollout（取樣動作）     最終價值 ≈ 800
回測（deterministic 平均）   最終價值 = 464,769     差 580 倍
```

**PPO 花 300k 步在最佳化一個你不會部署的策略。** 而 `log_std` 幾乎不動（1460 次更新只從 1.0 降到 0.92），不會自我修正。

```python
policy_kwargs={"log_std_init": -2.0}    # std = 0.135 → 噪音 0.68
```

修正前後（config B，有手續費）：

```
修正前   PPO  +7.7%  vs  B&H +86.4%    →  -78.7pp
修正後   PPO +83.0%  vs  B&H +86.4%    →   -3.5pp
```

### 5. 過擬合偵測（8 分）

**三段切分**：train 2011-2018 / **valid 2019** / test 2020-2022。

`ValidationCallback` 每 20k 步在 train 和 valid 兩邊跑 deterministic rollout，記錄六個指標，並存下驗證最佳的 checkpoint。

實際抓到的訊號（config B）：

```
 timesteps  train_reward  valid_reward  train_excess  valid_excess
    220000      0.0589       0.1052        0.0087       -0.0129
    260000      0.0699       0.0935        0.0197       -0.0247
    300000      0.0766       0.0782        0.0265       -0.0399
                   ↑            ↓             ↑            ↓
```

**train 持續上升、valid 持續下降** — 教科書等級的過擬合。最佳 checkpoint 在 200k 步，之後的 100k 步純粹在破壞泛化。

**要講的判讀重點**：

- 看 `train_excess` / `valid_excess`（各自減掉同期間的 B&H），**不要看 `gap`** — 後者被兩段期間的市場差異汙染（2019 對這些股票是好年，valid reward 天生就高）
- reward 一律換算成**每日**，否則 1980 天和 248 天不能比
- `explained_variance` 是**訓練集**上的，0.97 這麼高配合驗證下降，正是記憶的證據

**為什麼會記憶**：

```
每個 episode = 同一條路徑，順序固定，約 1940 步
300,000 步 / 1940 ≈ 155 次重播
```

同樣的 (狀態 → 答案) 配對看 155 次，每個 50 日視窗幾乎是唯一指紋。等於監督式學習裡「一個樣本訓練 155 個 epoch」。

### 6. 網路架構（6 分）

`MultiInputPolicy` 預設把 `(3,10,50)` 攤平成 1500 維 → 第一層約 97k 參數，而訓練資料只有 ~2000 個不重複的日子。

`EIIEExtractor` 把 EIIE 的卷積搬過來：kernel `(1, k_size)` **沿時間卷積、在資產軸共享權重**（這就是 "ensemble of identical independent evaluators"）。

```
1511 維 → 211 維，第一層參數 ~97k → ~13.5k
```

`ARCH = "flat"` / `"eiie"` 可切換對照。

#### EIIEExtractor 是不是原本的 EIIE？—— 卷積是，輸出頭不是

這點要講清楚，否則會被誤解成完整移植。

**相同的部分（實測驗證）：**

```
原 EIIE  卷積層參數 = 1,960
EIIEExtractor 卷積層 = 1,960     ← 完全一致
```

兩層卷積逐一對照：

| | 原 `architectures.py` 的 `EIIE` | `EIIEExtractor` |
|---|---|---|
| 第一層 | `Conv2d(3→2, kernel=(1,3))` + ReLU | 相同 |
| 第二層 | `Conv2d(2→20, kernel=(1,48))` + ReLU | 相同 |
| `n_size` 公式 | `time_window - k_size + 1` | 相同 |

**不同的部分 —— 輸出頭：**

```
原 EIIE:
  conv 輸出 (B, 20, 10, 1)
    → cat(last_action 各資產自己那一格)  → (B, 21, 10, 1)
    → Conv2d(21→1, kernel=(1,1))         → (B, 1, 10, 1)   ← 10 檔共用同一組 21 個權重
    → cat(cash_bias = 0)                 → (B, 1, 11, 1)
    → softmax                            → 合法權重，直接餵給環境

EIIEExtractor + SB3:
  conv 輸出 (B, 20, 10, 1)
    → Flatten                            → (B, 200)        ← 資產維度在這裡被攤平
    → cat(last_action 全部 11 格)         → (B, 211)
    → SB3 MlpExtractor(211→64→64)
    → action_net Linear(64→11)           → 每檔資產走不同的 dense 權重
    → 高斯取樣 → ×5 → 環境內 softmax
```

**核心差異：置換等變性（permutation equivariance）**

原 EIIE 的 `1×1` 卷積讓**每檔資產用完全相同的一組權重**產生自己的分數 ——
把 10 檔股票的順序打亂，輸出權重也只是跟著換位置。這是 EIIE 論文的核心設計。

`EIIEExtractor` 在 `Flatten` 之後就失去這個性質：資產對稱性只保留在**卷積階段**，
dense head 對每檔資產是不同的參數。

**為什麼還是這樣做**

要完整保留就得自訂 SB3 的 policy class（`ActorCriticPolicy` 的
`action_net` / `value_net`），工程量大很多，而且 SB3 的高斯策略本來就假設
動作維度彼此獨立、有各自的 `log_std`，跟 EIIE 的 `1×1` 共享頭在設計上有張力。

現在的版本已經拿到主要好處（**參數量降 50 倍 + 卷積階段的結構先驗**），
但要誠實說明它是**部分移植**，不是完整的 EIIE。

另外兩個小差異：

- 原 EIIE 把 **cash 的 logit 固定為 0**（`cash_bias = zeros`），只學風險資產的相對分數；
  我們的版本 11 個 logit 都自由學習。
- 原 EIIE **內部就 softmax**，輸出已是合法權重 → 走環境的 passthrough 分支；
  我們的版本輸出未正規化的動作 → 走環境的 softmax 分支（所以才需要 `action_scale`）。

### 6.5 PPO 是不是直接用 sb3？—— 是，完全未修改

```python
from stable_baselines3 import PPO          # 2.9.0
model = PPO("MultiInputPolicy", train_env, **PPO_KWARGS)
```

演算法、`MlpExtractor`、`Monitor`、`BaseCallback` **全部是 SB3 原生，一行未改**。
我們寫的只有「把 FinRL 環境接上 SB3」的黏合層：

| 元件 | 來源 |
|---|---|
| PPO 演算法、GAE、clipping、value function | **SB3 原生** |
| `MultiInputPolicy` / `CombinedExtractor` / `MlpExtractor` | **SB3 原生** |
| `Monitor` / `BaseCallback` / `CallbackList` | **SB3 原生** |
| `PortfolioOptimizationEnv` | **FinRL 原生，未修改** |
| `PortfolioOptimizationGymnasiumEnv` | 我們寫的 |
| `MultiPathEnv` | 我們寫的 |
| `EIIEExtractor` | 我們寫的（卷積照抄原 EIIE） |
| `PortfolioTrainingLogger` / `ValidationCallback` | 我們寫的 |

**沒有用 FinRL 內建的 `finrl/agents/stablebaselines3/models.py`**，原因：

- 它的 `DRL_prediction()` 呼叫 `save_asset_memory()` / `save_action_memory()`，
  這兩個方法只存在於 `StockTradingEnv`，投組環境沒有 → 直接 `AttributeError`
- 它的 `get_sb_env()` 會撞上 gymnasium 相容問題
- 預設 `PPO_PARAMS` 帶 `ent_coef=0.01`，而這正是我們診斷出讓策略不收斂的設定
- 它的 `TensorboardCallback` 只記單步 reward，對投組問題幫助有限

底層都是同一個 SB3 PPO，直接呼叫更透明，每個超參數都看得見。

**所以整個 pipeline 只有兩處是「非標準」的**，都在環境介面層，且都有實測理由：

1. `action_space_mode="symmetric"` + `action_scale=5` —— 繞開 softmax 動態範圍限制
2. `reward_scaling=100` —— 對數報酬太小，advantage 會被 critic 誤差淹沒

演算法本身沒有任何客製。

### 7. 結果與詮釋（5 分）

```
                    A_no_fee    B_with_fee
PPO  OOS 2020-2022   +92.51%      +82.95%
B&H  OOS 2020-2022   +90.47%      +86.41%
                    ─────────    ─────────
                     +2.04pp      -3.46pp
```

**成本帳**：

| | 手續費造成的損失 |
|---|---|
| PPO | -9.56pp |
| B&H | -4.06pp |
| PPO 多付 | **5.50pp** |

零成本下的優勢只有 2.04pp — **找到的 alpha 小於多付的交易成本**。

**但有一個一致的優點**（config B）：

```
最大回撤     PPO       B&H          Sharpe   PPO    B&H
train     -36.68%   -47.84%  ✅     2020    1.73   1.72  ✅
valid     -10.46%   -11.62%  ✅     2021   -0.01  -0.16  ✅
2020      -23.41%   -25.06%  ✅     2022    0.94   0.84  ✅
2021      -14.33%   -17.28%  ✅
2022      -15.59%   -16.61%  ✅
```

五個期間回撤全部較低，其中四個是未見過的資料；三個測試年 Sharpe 全勝。

**結論的正確講法**：

> 訓練是成功的，agent 學到一個真實且可泛化的技能 — **降低波動**。但那個技能不足以在扣掉交易成本後產生超額報酬。

而這正是 reward 設計會鼓勵的（log 的凹性懲罰波動）。**用總報酬評判它，剛好評到它最弱的那一面。**

**必須聲明的保留**：以上全部是**單一 seed**。先前觀察到同一份資料上連續 episode 的結果可差 5 倍，這些結論需要多 seed 才能定論。

---

## 三、多組生成資料從哪裡接進去

### 為什麼這是對症下藥

過擬合的根源是「1 條路徑重播 155 次」。多路徑直接消除這個條件：

```
現在      1 條路徑 × 155 次重播     → 可以背下來
改成     50 條路徑 ×   3 次重播     → 背不起來，只能學通用型態
```

agent 沒辦法再記住「第 437 天買 PETR4」，因為每個 episode 的第 437 天都不一樣。

### 檔案放哪裡

生成器輸出一批 CSV，放在專案下的 `data/synthetic/`（名稱不重要，但要跟真實資料分開）：

```
FinRL/
├── examples/
│   ├── portfolio_raw_frozen.csv          真實資料（凍結）
│   └── FinRL_PortfolioOptimization_PPO_Demo.ipynb
└── data/
    └── synthetic/
        ├── path_000.csv                  每個檔案 = 一條完整路徑
        ├── path_001.csv                  欄位: date,tic,close,high,low
        ├── ...
        └── path_049.csv
```

每個檔案是**一條完整的、跨越整個訓練期的路徑**（約 1980 天 × 10 檔 = 19,800 列），
不是把一條路徑切段。50 條路徑約 100 MB，如果嫌大可以存 `.parquet` 或只存
對數報酬（`(N, 3, T)` 的 `.npy`）在載入時再還原成價格。

### 接進去的程式碼

在 notebook §6（建立 `environment` 的那個 cell）之前插入：

```python
from pathlib import Path
from finrl.meta.env_portfolio_optimization.env_portfolio_multipath import MultiPathEnv

SYNTH_DIR = Path("../data/synthetic")     # notebook 在 examples/ 底下
USE_SYNTHETIC = True                       # 設 False 就退回單一真實路徑

if USE_SYNTHETIC:
    synth_dfs = []
    for p in sorted(SYNTH_DIR.glob("path_*.csv")):
        d = pd.read_csv(p)
        # 跟真實資料走完全相同的正規化流程
        d = GroupByScaler(by="tic", scaler=MaxAbsScaler,
                          columns=["close", "high", "low"]).fit_transform(d)[COLS]
        synth_dfs.append(d)

    dfs = [df_train] + synth_dfs
    n = len(synth_dfs)
    environment = MultiPathEnv.from_dataframes(
        dfs,
        seed=SEED,
        weights=[0.3] + [0.7 / n] * n,     # 真實路徑佔 30%；省略則均等
        **ENV_KWARGS,
    )
    print(f"training on {len(dfs)} paths (1 real + {n} synthetic)")
else:
    environment = PortfolioOptimizationGymnasiumEnv(df_train, **ENV_KWARGS)
```

**其餘完全不用改。** `Monitor(environment)`、`PPO(...)`、`ValidationCallback`、
回測 —— 全部照舊，因為：

- `MultiPathEnv` 的 observation/action space 跟單一環境相同
- `ValidationCallback` 和回測**自己建立環境**，用的是 `df_train` / `df_valid` / 測試年份，
  仍然是真實資料

### 為什麼合成資料也要各自做 MaxAbs 正規化

環境算報酬用的是比值 `close_t / close_{t-1}`，縮放常數會約掉，**所以正規化不影響績效**。
但網路吃的是絕對數值，如果合成路徑的尺度跟真實路徑差很多，agent 會學到
「數值大小」這個沒有意義的特徵。各自縮放到 `[0, 1]` 就對齊了。

（合成資料用自己的 max 來 fit 沒有洩漏問題 —— 它本來就是憑空生成的，
不存在「未來資訊」。）

### 混合比例怎麼選

`weights` 控制真實路徑被抽到的機率：

| 設定 | 效果 |
|---|---|
| 省略（均等） | 真實路徑佔 `1/51`，幾乎純合成訓練 |
| `[0.3] + [0.7/n]*n` | 真實佔 30%，建議起點 |
| `[0.5] + [0.5/n]*n` | 各半，較保守 |

保留一定比例的真實路徑，是為了在 diffusion 沒學好時仍有錨點。
如果生成品質有信心，可以降到 10% 或完全不放。

### 生成資料要長什麼樣

跟真實資料完全相同的長格式：

```csv
date,tic,close,high,low
2011-01-03,SYN_00,76.325110,76.613292,76.053951
2011-01-03,SYN_01,75.090949,75.378205,74.589663
```

**硬性約束**（`make_synthetic_portfolio_data.py` 的 `validate()` 會檢查）：

1. 每個 tic 的日期集合完全相同 — 否則疊張量會 shape error
2. 所有數值 > 0 — 報酬是除法
3. `low ≤ close ≤ high`
4. 天數 > `time_window`
5. 各路徑的資產數與 `time_window` 必須一致

### 建議的生成方式：3 channel 表示法

**不要讓生成模型直接輸出 high/low** — 三條線獨立生成必然出現 `high < close`，事後 clip 會破壞分佈。

改成生成三個無界量：

| channel | 內容 | 為什麼 |
|---|---|---|
| 0 | `log(C_t / C_{t-1})` | 收盤對數報酬 |
| 1 | `log( log(H_t / C_t) + eps )` | 上影線，外層 log 解除非負約束 |
| 2 | `log( log(C_t / L_t) + eps )` | 下影線 |

反轉換時 `low ≤ close ≤ high` **數學上自動成立**：

```
C_t = C_0 * exp(cumsum(ch0))        只有這條沿時間累積
u_t = exp(ch1) - eps  >= 0
H_t = C_t * exp(u_t)                掛在當天的 C 上，不會漂移
L_t = C_t * exp(-d_t)
```

**關鍵：channel 1、2 是「同一天內 high/low 離 close 多遠」，不是「high 的報酬率」。** 後者三條線各自累積，跑到後期必然交叉。

**必須聯合生成 `(N, 3, T)`** — 資產之間的相關結構是投組最佳化的全部價值所在，獨立生成等於把它丟掉。同理 channel 之間也有相關（大跌日下影線長）。

### 訓練資料必須是除權息調整後的

`YahooDownloader` 已經做了。用未調整資料訓練生成模型，模型會把「分割日腰斬」當成真實市場現象學起來，然後隨機吐出假的 -50% 崩跌。

（合成資料本身不需要「再調整」— 它沒有除權息事件。關鍵在上游用什麼資料訓練生成模型。）

### 驗證與測試必須維持真實資料

`ValidationCallback` 和回測都建立自己的環境，本來就用真實資料，不需要改。

**這點不能妥協** — 否則你只是在驗證「模型學會了生成器的分佈」，不是「模型能賺錢」。

### 成功的判準

不是「訓練 reward 變高」，而是：

```
train_reward                 ↓  下降（問題變難，本來就該降）
valid_reward                 →  持平或上升
train_excess - valid_excess  ↓  落差縮小        ← 這才是重點
valid_excess                 ↑  升破 0          ← 真正的成功
```

---

## 四、已知限制（要主動說明）

1. **單一 seed** — 所有數字都缺統計支撐，需要多 seed 看分佈
2. **只有一條真實路徑** — 歷史只發生過一次；這正是多路徑訓練要解決的
3. **驗證期只有一年** — 2019 是單一市場情境，選出的 checkpoint 可能只適合那個情境
4. **`clip_fraction ≈ 0.42`** — 正常應 <0.2，代表更新步幅過大，應加 `target_kl=0.02` 或降 `n_epochs`
5. **手續費對 EIIE 版本的梯度無貢獻** — 原 `PolicyGradient` 把 `trf_mu` detach 成常數；PPO 版本沒這問題（reward 直接含成本）
6. **不含市場衝擊** — 假設下單不影響價格
7. **`TOTAL_TIMESTEPS=300k` 偏長** — 兩個 config 都在 180k–200k 後開始過擬合，可降到 200k

---

## 五、常見問題

**Q: 這是標準 RL 嗎？跟原 demo 的 `PolicyGradient` 差在哪？**

兩者都是 RL，但梯度估計方式不同：

| | `PolicyGradient`（原 demo） | PPO（本 notebook） |
|---|---|---|
| 梯度 | **直接微分 reward**（pathwise） | 取樣估計 |
| 前提 | 環境已知且可微 | 環境可以是黑盒 |
| critic | 無 | 有 |
| 探索 | 無 | 高斯噪聲 |
| 樣本效率 | 高 | 低 |

原 demo 能直接微分，是因為 `reward = log(w·y)` 對 `w` 可微，且 `y`（價格變動）是外生常數。股票交易環境做不到，因為買賣整數股數是不可微的階梯函數。

**Q: 為什麼 `reward_scaling` 在兩邊行為不同？**

`PolicyGradient` 把環境的 reward 丟掉、從 `info` 重算，所以 `reward_scaling` **完全無效**。PPO 直接吃 `step()` 的 reward，所以**有效**，而且對數報酬只有 ±0.02，不放大會被 critic 誤差淹沒。

**Q: 為什麼不用 FinRL 內建的 `DRLAgent`？**

它是為 stock trading 環境寫的：`DRL_prediction()` 呼叫 `save_asset_memory()`（投組環境沒有這個方法）、預設 `PPO_PARAMS` 帶 `ent_coef=0.01`（會讓策略不收斂）。底層都是同一個 SB3 PPO，直接呼叫更透明。

**Q: 修改了原始碼嗎？**

沒有。`PortfolioOptimizationEnv` 一行都沒動，新功能都在子類別裡。
