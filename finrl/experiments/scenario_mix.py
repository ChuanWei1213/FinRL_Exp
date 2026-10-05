"""10-stock + cash PPO environment mixing real episodes with TimeDiff stress scenarios.

Episode layout (A and B groups identical; only the source of the last 21 days differs):

    |<-- 50 obs days -->|<-- k real days -->|<-- 21 future days -->|
       observe only          trade               trade (real or TimeDiff)
    k ~ Uniform{0..k_max}, drawn at every reset (also for real episodes).

Design rules:
  * Start weights = 1/11 (cash + 10 stocks); the initial position costs nothing during
    training. Evaluation charges the build-up cost once.
  * The episode terminates right after the last real trade (FinRL's extra dummy step,
    which repeats the previous reward, is removed).
  * Observation = 50-day close/high/low window divided by each stock's latest close in
    the window, so training episodes and the 3-year continuous test look alike.
  * Scenario labels / real-vs-synthetic flags never enter the observation.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

FEATURES = ["close", "high", "low"]


# ─────────────────────────────── data ───────────────────────────────
class MarketData:
    """Wide arrays [T, N] for close/high/low from TimeDiff's long prices.csv."""

    def __init__(self, path_or_frame, tickers=None):
        df = pd.read_csv(path_or_frame, parse_dates=["date"]) if not isinstance(
            path_or_frame, pd.DataFrame) else path_or_frame.copy()
        tickers = list(tickers) if tickers is not None else sorted(df.tic.unique())
        self.tickers = tickers
        piv = {c: df.pivot(index="date", columns="tic", values=c)[tickers] for c in FEATURES}
        if any(p.isna().any().any() for p in piv.values()):
            raise ValueError("missing prices; every date must have every ticker")
        self.dates = pd.DatetimeIndex(piv["close"].index)
        self.close, self.high, self.low = (piv[c].to_numpy(float) for c in FEATURES)

    def idx(self, date, side="left"):
        return int(self.dates.searchsorted(pd.Timestamp(date), side=side))

    def last_idx(self, date):
        """Index of the last trading day <= date."""
        return self.idx(date, side="right") - 1


class SyntheticPaths:
    """paths.npz from dow10_cond_experiment.py generate --mode rl."""

    def __init__(self, path, market: MarketData):
        z = np.load(path, allow_pickle=False)
        if list(z["tickers"]) != market.tickers:
            raise ValueError("ticker order differs between paths.npz and prices")
        self.close, self.high, self.low = z["close"], z["high"], z["low"]   # [n, 21, N]
        self.anchor_idx = np.array([market.idx(str(d)) for d in z["anchor_dates"]])
        if (market.dates[self.anchor_idx] != pd.to_datetime([str(d) for d in z["anchor_dates"]])).any():
            raise ValueError("anchor dates not found in market data")
        self.scenario = z["scenario"]
        # Anchor close in paths must equal the real close (continuity check).
        first = self.close[:, 0, :] / np.exp(z["raw_features"][:, 0, ::3])
        if not np.allclose(first, market.close[self.anchor_idx], rtol=1e-4):
            raise ValueError("synthetic paths do not continue from the real anchor close")

    def __len__(self):
        return len(self.close)


def episode_frame(dates, close, high, low, tickers):
    t, n = close.shape
    return pd.DataFrame({
        "date": np.repeat(pd.DatetimeIndex(dates).to_numpy(), n),
        "tic": np.tile(tickers, t),
        "close": close.ravel(), "high": high.ravel(), "low": low.ravel(),
    })


def env_kwargs(commission, obs_days=50, initial_amount=100_000, reward_scaling=100.0):
    """Same interface choices as synthetic_vs_real.environment_kwargs."""
    return {"initial_amount": initial_amount, "time_window": obs_days, "features": list(FEATURES),
            "normalize_df": None, "reward_scaling": reward_scaling,
            "action_space_mode": "symmetric", "action_scale": 5.0, "return_last_action": True,
            "plot_on_terminal": False, "cwd": ".", "comission_fee_pct": float(commission)}


# ─────────────────────────────── env ───────────────────────────────
_FAST_ENV_CLASS = None


def _fast_env_class():
    """PortfolioOptimizationGymnasiumEnv with the per-step pandas filtering replaced by
    precomputed numpy arrays. Trading math (fees, rewards, weights) is inherited
    unchanged; only how the state window and price variation are looked up differs.
    This alone makes a step ~50x faster."""
    global _FAST_ENV_CLASS
    if _FAST_ENV_CLASS is not None:
        return _FAST_ENV_CLASS
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
        PortfolioOptimizationGymnasiumEnv)

    class FastPortfolioEnv(PortfolioOptimizationGymnasiumEnv):
        def __init__(self, df, **kw):
            super().__init__(df, **kw)
            tcol, kcol = self._time_column, self._tic_column
            T, N = len(self._sorted_times), len(self._tic_list)
            cube = np.empty((len(self._features), N, T), dtype=self._df[self._features[0]].dtype)
            pv = np.empty((T, N), dtype=self._df_price_variation[self._valuation_feature].dtype)
            dpv = self._df_price_variation
            for j, tic in enumerate(self._tic_list):
                sub = self._df[self._df[kcol] == tic].sort_values(tcol)
                cube[:, j, :] = sub[self._features].to_numpy().T
                pv[:, j] = dpv[dpv[kcol] == tic].sort_values(tcol)[self._valuation_feature].to_numpy()
            self._cube, self._pv = cube, pv

        def _get_state_and_info_from_time_index(self, time_index):
            if not hasattr(self, "_cube"):          # during parent __init__
                return super()._get_state_and_info_from_time_index(time_index)
            W = self._time_window
            state = self._cube[:, :, time_index - W + 1:time_index + 1]
            self._price_variation = np.insert(self._pv[time_index], 0, 1)
            self._data = None
            info = {"tics": self._tic_list,
                    "start_time": self._sorted_times[time_index - (W - 1)],
                    "start_time_index": time_index - (W - 1),
                    "end_time": self._sorted_times[time_index], "end_time_index": time_index,
                    "data": None, "price_variation": self._price_variation}
            return self._standardize_state(state), info

    _FAST_ENV_CLASS = FastPortfolioEnv
    return FastPortfolioEnv


class _InnerEpisode:
    """One (fast) portfolio env over a fixed frame, started at 1/(N+1)."""

    def __init__(self, frame, n_assets, commission, obs_days, fast=True):
        if fast:
            cls = _fast_env_class()
        else:
            from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
                PortfolioOptimizationGymnasiumEnv as cls)
        self.env = cls(frame, **env_kwargs(commission, obs_days))
        self.w0 = np.full(n_assets + 1, 1.0 / (n_assets + 1), dtype=np.float32)

    def reset(self):
        e = self.env
        obs, info = e.reset()
        # Start already invested at 1/(N+1) without paying for it.
        e._actions_memory = [self.w0.copy()]
        e._final_weights = [self.w0.copy()]
        obs = dict(obs)
        obs["last_action"] = self.w0.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        e = self.env
        # Terminate right after the last real trade: no dummy step with a repeated reward.
        if e._time_index >= len(e._sorted_times) - 1:
            terminated = True
        return obs, reward, terminated, truncated, info


def normalize_obs(obs):
    state = np.asarray(obs["state"], dtype=np.float32)                 # (3, N, W)
    last_close = state[0, :, -1][None, :, None]
    return {"state": (state / last_close).astype(np.float32),
            "last_action": np.asarray(obs["last_action"], dtype=np.float32)}


class ScenarioMixEnv(gym.Env):
    """Each reset draws k, then a real episode (prob 1-p) or a TimeDiff path (prob p)."""

    metadata = {"render_modes": []}

    def __init__(self, market: MarketData, synthetic: SyntheticPaths | None, *, p_synthetic: float,
                 data_start: str, data_end: str, commission: float, obs_days: int = 50,
                 horizon: int = 21, k_max: int = 21, seed: int = 0):
        super().__init__()
        if p_synthetic and synthetic is None:
            raise ValueError("p_synthetic > 0 needs synthetic paths")
        if not 0 <= p_synthetic < 1:
            raise ValueError("p_synthetic must be in [0, 1)")
        self.m, self.syn, self.p = market, synthetic, float(p_synthetic)
        self.obs_days, self.horizon, self.k_max = obs_days, horizon, k_max
        self.commission = commission
        self.lo = market.idx(data_start)            # first row any episode may touch
        self.hi = market.last_idx(data_end)         # last real row any episode may touch
        self.rng = np.random.default_rng(seed)
        n = len(market.tickers)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (len(FEATURES), n, obs_days), np.float32),
            "last_action": spaces.Box(0.0, 1.0, (n + 1,), np.float32)})
        self.action_space = spaces.Box(-1.0, 1.0, (n + 1,), np.float32)
        # Validity tables per k.
        self._real_starts = {k: np.arange(self.lo + obs_days - 1, self.hi - k - horizon + 1)
                             for k in range(k_max + 1)}
        if synthetic is not None:
            a = synthetic.anchor_idx
            if (a > self.hi).any():
                raise ValueError("synthetic anchors after data_end would leak later history")
            self._syn_ok = {k: np.flatnonzero(a - k - obs_days + 1 >= self.lo) for k in range(k_max + 1)}
        for k, s in self._real_starts.items():
            if len(s) == 0:
                raise ValueError(f"no real episode fits for k={k}")
        self.last_source = None
        self._inner = None

    def _build(self):
        k = int(self.rng.integers(0, self.k_max + 1))
        m, W = self.m, self.obs_days
        if self.p and self.rng.random() < self.p:
            j = int(self.rng.choice(self._syn_ok[k]))
            a = int(self.syn.anchor_idx[j])
            s = a - k
            rows = slice(s - W + 1, a + 1)
            fut_dates = pd.bdate_range(m.dates[a] + pd.offsets.BDay(1), periods=self.horizon)
            dates = np.r_[m.dates[rows].to_numpy(), fut_dates.to_numpy()]
            close = np.vstack([m.close[rows], self.syn.close[j]])
            high = np.vstack([m.high[rows], self.syn.high[j]])
            low = np.vstack([m.low[rows], self.syn.low[j]])
            self.last_source = ("synthetic", int(self.syn.scenario[j]), k)
        else:
            s = int(self.rng.choice(self._real_starts[k]))
            rows = slice(s - W + 1, s + k + self.horizon + 1)
            dates, close, high, low = m.dates[rows], m.close[rows], m.high[rows], m.low[rows]
            self.last_source = ("real", None, k)
        frame = episode_frame(dates, close, high, low, m.tickers)
        return _InnerEpisode(frame, len(m.tickers), self.commission, W)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._inner = self._build()
        obs, info = self._inner.reset()
        return normalize_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        return normalize_obs(obs), float(reward), bool(terminated), bool(truncated), info


# ─────────────────────────────── evaluation ───────────────────────────────
def run_continuous(policy, market: MarketData, start, end, commission, obs_days=50):
    """Deterministic rollout on real data from `start` to `end`, starting at 1/(N+1).

    The build-up cost (cash -> 10/11 in stocks) is charged once, up front.
    Returns daily DataFrame: date, value, cash_weight, turnover.
    """
    s, e = market.idx(start), market.last_idx(end)
    rows = slice(s - obs_days, e + 1)            # 50 obs days end the day before `start`
    frame = episode_frame(market.dates[rows], market.close[rows], market.high[rows],
                          market.low[rows], market.tickers)
    inner = _InnerEpisode(frame, len(market.tickers), commission, obs_days)
    obs, _ = inner.reset()
    obs = normalize_obs(obs)
    done = False
    while not done:
        obs, _, done, _, _ = inner.step(policy(obs))
        obs = normalize_obs(obs)
    env = inner.env
    n = len(market.tickers)
    entry = 1 - commission * n / (n + 1)
    values = np.asarray(env._asset_memory["final"], float) / env._initial_amount * entry   # start = 1
    acts = np.asarray(env._actions_memory, float)
    finals = np.asarray(env._final_weights, float)
    turnover = np.r_[n / (n + 1), np.abs(acts[1:] - finals[:-1]).sum(axis=1) / 2]
    dates = market.dates[s - 1:e + 1]            # value[0] = start-of-test (after entry cost)
    return pd.DataFrame({"date": dates, "value": values, "cash_weight": finals[:, 0],
                         "turnover": turnover})


def buy_and_hold(market: MarketData, start, end, commission, include_cash=True):
    s, e = market.idx(start), market.last_idx(end)
    n = len(market.tickers)
    w_stock = 1 / (n + 1) if include_cash else 1 / n
    entry = 1 - commission * w_stock * n
    rel = market.close[s - 1:e + 1] / market.close[s - 1]
    value = ((1 - w_stock * n) + (w_stock * rel).sum(axis=1)) * entry
    return pd.DataFrame({"date": market.dates[s - 1:e + 1], "value": value})


def period_metrics(daily: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    out = []
    d = daily.set_index("date")
    for name, (a, b) in periods.items():
        seg = d.loc[:b]
        prev = seg.loc[:pd.Timestamp(a) - pd.Timedelta(days=1)]
        base = prev.value.iloc[-1] if len(prev) else seg.value.iloc[0]
        v = seg.loc[a:].value
        path = np.r_[base, v.to_numpy()]
        r21 = path[21:] / path[:-21] - 1 if len(path) > 21 else np.array([path[-1] / path[0] - 1])
        logr = np.diff(np.log(path))
        row = {"period": name, "return": path[-1] / path[0] - 1,
               "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
               "worst_21d_return": float(r21.min()),
               "ann_vol": float(logr.std(ddof=1) * np.sqrt(252)),
               "sharpe_0rf": float(logr.mean() / logr.std(ddof=1) * np.sqrt(252)) if logr.std() > 0 else np.nan}
        if "cash_weight" in seg:
            row["mean_stock_weight"] = float(1 - seg.loc[a:].cash_weight.mean())
            row["mean_daily_turnover"] = float(seg.loc[a:].turnover.mean())
        out.append(row)
    return pd.DataFrame(out)
