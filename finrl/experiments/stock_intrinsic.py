from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

from finrl.experiments.notebook_utils import performance_metrics_from_values
from finrl.experiments.notebook_utils import set_global_seed
from finrl.experiments.notebook_utils import TqdmTrainingCallback
from finrl.experiments.notebook_utils import validate_periods
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_intrinsic import (
    IntrinsicRewardStockTradingEnv,
)
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.rewards import IntrinsicRewardConfig


DEFAULT_TECH_INDICATORS = (
    "macd",
    "rsi_30",
    "close_30_sma",
    "close_60_sma",
)

VARIANT_WEIGHTS = {
    "baseline": (0.0, 0.0),
    "surprise": (0.05, 0.0),
    "dejavu": (0.0, 0.05),
    "combined": (0.05, 0.05),
}

EVALUATION_METRICS = (
    "final_value",
    "cumulative_return",
    "cagr",
    "sharpe",
    "annualized_volatility",
    "max_drawdown",
)

TRAIN_CACHE_SCHEMA = "stock_intrinsic_v3"
MAX_INTRINSIC_REPLAY_CAPACITY = 100_000


@dataclass
class VariantRun:
    variant: str
    seed: int
    model: PPO
    environment: StockTradingEnv
    intrinsic_log: pd.DataFrame
    validation_log: pd.DataFrame
    best_timestep: int
    cache_hit: bool = False
    cache_key: str = ""


@dataclass(frozen=True)
class VariantTrainingPlan:
    variant: str
    seed: int
    cache_key: str
    cache_payload: dict
    cached_run_dir: Path | None
    cache_hit: bool


class IntrinsicMetricsCallback(BaseCallback):
    """Collect training diagnostics and periodic external-reward validation."""

    def __init__(
        self,
        log_every: int = 16,
        training_frame: pd.DataFrame | None = None,
        validation_frame: pd.DataFrame | None = None,
        validation_env_kwargs: dict | None = None,
        eval_freq: int | None = None,
    ):
        super().__init__()
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        if eval_freq is not None and eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if (validation_frame is None) != (validation_env_kwargs is None):
            raise ValueError(
                "validation_frame and validation_env_kwargs must be provided together"
            )
        if validation_frame is not None and eval_freq is None:
            raise ValueError("eval_freq is required when validation is enabled")
        if validation_frame is not None and training_frame is None:
            raise ValueError("training_frame is required when validation is enabled")
        self.log_every = log_every
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.validation_env_kwargs = validation_env_kwargs
        self.eval_freq = eval_freq
        self.records: list[dict[str, float | int | bool]] = []
        self.validation_records: list[dict[str, float | int | bool]] = []
        self.best_score = -np.inf
        self.best_timestep = 0
        self.best_policy_state: dict | None = None

    def _on_training_start(self) -> None:
        if self.validation_frame is not None:
            self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            infos = self.locals.get("infos", [])
            rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
            reward_train = float(rewards.mean()) if rewards.size else np.nan
            numeric_keys = (
                "reward_ext",
                "reward_intrinsic",
                "reward_surprise_raw",
                "reward_surprise",
                "reward_dejavu_raw",
                "reward_dejavu",
                "intrinsic_eta",
                "intrinsic_time_decay",
                "intrinsic_volatility_factor",
                "intrinsic_surprise_loss",
                "intrinsic_dejavu_loss",
            )
            record: dict[str, float | int | bool] = {
                "timesteps": int(self.num_timesteps),
                "reward_train": reward_train,
            }
            for key in numeric_keys:
                values = [float(info[key]) for info in infos if key in info]
                default = reward_train if key == "reward_ext" else 0.0
                record[key] = float(np.mean(values)) if values else default
            record["intrinsic_warmup"] = bool(
                any(info.get("intrinsic_warmup", False) for info in infos)
            )
            logger_values = self.model.logger.name_to_value
            for logger_key, output_key in (
                ("rollout/ep_rew_mean", "episode_reward_mean"),
                ("train/explained_variance", "explained_variance"),
                ("train/value_loss", "value_loss"),
                ("train/entropy_loss", "entropy_loss"),
                ("train/approx_kl", "approx_kl"),
            ):
                value = logger_values.get(logger_key, np.nan)
                record[output_key] = float(value)
            self.records.append(record)

        should_validate = (
            self.validation_frame is not None
            and self.eval_freq is not None
            and self.num_timesteps % self.eval_freq == 0
        )
        if should_validate:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        train_result = rollout_stock_model(
            self.training_frame,
            self.model,
            self.validation_env_kwargs,
        )
        validation_result = rollout_stock_model(
            self.validation_frame,
            self.model,
            self.validation_env_kwargs,
        )
        validation_metrics = validation_result["metrics"]
        score = float(validation_metrics["reward_per_day"])
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_timestep = int(self.num_timesteps)
            self.best_policy_state = copy.deepcopy(self.model.policy.state_dict())
        self.validation_records.append(
            {
                "timesteps": int(self.num_timesteps),
                "train_reward_mean": float(train_result["metrics"]["reward_per_day"]),
                "train_reward_std": np.nan,
                "real_valid_reward": score,
                "real_valid_turnover": float(validation_metrics["turnover"]),
                "real_valid_max_weight": float(validation_result["max_weight"]),
                "is_best": is_best,
                **validation_metrics,
            }
        )


def _cache_serializable(value):
    if isinstance(value, dict):
        return {
            str(key): _cache_serializable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_cache_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def stock_frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash dataframe values, column order, and dtypes for cache validation."""
    ordered = frame.sort_values(["date", "tic"], kind="mergesort").reset_index(
        drop=True
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(list(ordered.columns)).encode())
    dtypes = json.dumps(
        ordered.dtypes.astype(str).to_dict(),
        sort_keys=True,
    )
    digest.update(dtypes.encode())
    digest.update(pd.util.hash_pandas_object(ordered, index=True).values.tobytes())
    return digest.hexdigest()


def stock_training_cache_key(
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    variant: str,
    total_timesteps: int,
    eval_freq: int | None,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict,
    ppo_kwargs: dict,
) -> tuple[str, dict]:
    """Return a semantic cache key and the auditable payload behind it."""
    payload = {
        "schema": TRAIN_CACHE_SCHEMA,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "initial_allocation": "cash",
        "train_fingerprint": stock_frame_fingerprint(train),
        "validation_fingerprint": (
            stock_frame_fingerprint(validation) if validation is not None else None
        ),
        "variant": variant,
        "variant_weights": VARIANT_WEIGHTS[variant],
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq) if eval_freq is not None else None,
        "seed": int(seed),
        "warmup_steps": int(warmup_steps),
        "intrinsic_reward": (
            None
            if variant == "baseline"
            else {
                "alpha": VARIANT_WEIGHTS[variant][0],
                "beta": VARIANT_WEIGHTS[variant][1],
                "batch_size": min(128, max(2, warmup_steps)),
                "replay_capacity": min(
                    MAX_INTRINSIC_REPLAY_CAPACITY,
                    max(10_000, total_timesteps),
                ),
                "device": str(ppo_kwargs.get("device", "auto")),
            }
        ),
        "environment": _cache_serializable(env_kwargs),
        "ppo": _cache_serializable(ppo_kwargs),
        "versions": {
            "stable_baselines3": sb3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _write_run_artifacts(
    run_dir: Path,
    model: PPO,
    environment: StockTradingEnv,
    training_log: pd.DataFrame,
    validation_log: pd.DataFrame,
    metadata: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "agent_ppo")
    if isinstance(environment, IntrinsicRewardStockTradingEnv):
        environment.save_intrinsic_state(run_dir / "intrinsic_reward.pt")
    training_log.to_csv(run_dir / "intrinsic_training_log.csv", index=False)
    validation_log.to_csv(run_dir / "validation_log.csv", index=False)
    with (run_dir / "cache_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_cached_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _cached_run_is_complete(run_dir: Path, variant: str, cache_key: str) -> bool:
    required = {
        "agent_ppo.zip",
        "intrinsic_training_log.csv",
        "validation_log.csv",
        "cache_metadata.json",
    }
    if variant != "baseline":
        required.add("intrinsic_reward.pt")
    if not all((run_dir / filename).is_file() for filename in required):
        return False
    try:
        with (run_dir / "cache_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    return bool(
        metadata.get("complete")
        and metadata.get("schema") == TRAIN_CACHE_SCHEMA
        and metadata.get("cache_key") == cache_key
    )


def plan_variant_training(
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    variant: str,
    total_timesteps: int,
    eval_freq: int | None,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict,
    ppo_kwargs: dict,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    force_retrain: bool = False,
) -> VariantTrainingPlan:
    """Check one semantic training cache entry without building an environment."""
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown variant {variant!r}")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if validation is not None and eval_freq is None:
        raise ValueError("eval_freq is required when validation is enabled")
    cache_key, cache_payload = stock_training_cache_key(
        train=train,
        validation=validation,
        variant=variant,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        seed=seed,
        warmup_steps=warmup_steps,
        env_kwargs=env_kwargs,
        ppo_kwargs=ppo_kwargs,
    )
    cached_run_dir = (
        Path(cache_dir) / cache_key if cache_dir is not None and use_cache else None
    )
    cache_hit = bool(
        not force_retrain
        and cached_run_dir is not None
        and _cached_run_is_complete(cached_run_dir, variant, cache_key)
    )
    return VariantTrainingPlan(
        variant=variant,
        seed=int(seed),
        cache_key=cache_key,
        cache_payload=cache_payload,
        cached_run_dir=cached_run_dir,
        cache_hit=cache_hit,
    )


def prepare_stock_splits(
    real_ohlcv: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    tech_indicators: Sequence[str] = DEFAULT_TECH_INDICATORS,
    feature_lookback_days: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leak-resistant technical features, then create FinRL-indexed splits."""
    validate_periods(
        (("train", train_start, train_end), ("test", test_start, test_end))
    )
    if feature_lookback_days <= 0:
        raise ValueError("feature_lookback_days must be positive")
    required = {"date", "tic", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(real_ohlcv.columns))
    if missing:
        raise ValueError(f"real_ohlcv is missing columns {missing}")

    frame = real_ohlcv.copy()
    dates = pd.to_datetime(frame["date"], errors="raise")
    feature_start = pd.Timestamp(train_start) - pd.Timedelta(days=feature_lookback_days)
    frame = frame[(dates >= feature_start) & (dates < pd.Timestamp(test_end))]
    frame = frame.sort_values(["date", "tic"], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=list(tech_indicators),
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    featured = engineer.preprocess_data(frame)
    feature_values = featured[list(tech_indicators)].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("engineered features contain NaN or infinity")

    train = data_split(featured, train_start, train_end)
    test = data_split(featured, test_start, test_end)
    if train.empty or test.empty:
        raise ValueError("train and test splits must both contain data")
    expected_tickers = sorted(featured["tic"].unique())
    for label, split in (("train", train), ("test", test)):
        actual_tickers = sorted(split["tic"].unique())
        if actual_tickers != expected_tickers:
            raise ValueError(
                f"{label}: expected tickers {expected_tickers}, got {actual_tickers}"
            )
    return train, test


def prepare_stock_periods(
    real_ohlcv: pd.DataFrame,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    tech_indicators: Sequence[str] = DEFAULT_TECH_INDICATORS,
    feature_lookback_days: int = 180,
) -> dict[str, pd.DataFrame]:
    """Create chronological train, validation, and test StockTrading frames."""
    validate_periods(
        (
            ("train", train_start, train_end),
            ("validation", validation_start, validation_end),
            ("test", test_start, test_end),
        )
    )
    if train_end != validation_start or validation_end != test_start:
        raise ValueError("train, validation, and test periods must be contiguous")
    if feature_lookback_days <= 0:
        raise ValueError("feature_lookback_days must be positive")

    required = {"date", "tic", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(real_ohlcv.columns))
    if missing:
        raise ValueError(f"real_ohlcv is missing columns {missing}")

    frame = real_ohlcv.copy()
    dates = pd.to_datetime(frame["date"], errors="raise")
    feature_start = pd.Timestamp(train_start) - pd.Timedelta(days=feature_lookback_days)
    frame = frame[(dates >= feature_start) & (dates < pd.Timestamp(test_end))]
    frame = frame.sort_values(["date", "tic"], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=list(tech_indicators),
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    featured = engineer.preprocess_data(frame)
    feature_values = featured[list(tech_indicators)].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("engineered features contain NaN or infinity")

    bounds = {
        "train": (train_start, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, test_end),
    }
    expected_tickers = sorted(real_ohlcv["tic"].astype(str).unique())
    periods = {}
    for label, (start, end) in bounds.items():
        split = data_split(featured, start, end)
        if split.empty:
            raise ValueError(f"{label} split must not be empty")
        actual_tickers = sorted(split["tic"].astype(str).unique())
        if actual_tickers != expected_tickers:
            raise ValueError(
                f"{label}: expected tickers {expected_tickers}, got {actual_tickers}"
            )
        periods[label] = split
    return periods


def stock_environment_kwargs(
    frame: pd.DataFrame,
    tech_indicators: Sequence[str] = DEFAULT_TECH_INDICATORS,
    initial_amount: int = 100_000,
    hmax: int = 100,
    transaction_cost: float = 0.001,
    reward_scaling: float = 1e-4,
) -> dict:
    tickers = sorted(frame["tic"].unique())
    stock_dimension = len(tickers)
    if stock_dimension == 0:
        raise ValueError("frame must contain at least one ticker")
    return {
        "stock_dim": stock_dimension,
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [transaction_cost] * stock_dimension,
        "sell_cost_pct": [transaction_cost] * stock_dimension,
        "reward_scaling": reward_scaling,
        "state_space": 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension,
        "action_space": stock_dimension,
        "tech_indicator_list": list(tech_indicators),
        "print_verbosity": 10_000,
    }


def build_training_environment(
    train: pd.DataFrame,
    variant: str,
    total_timesteps: int,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict,
    device: str = "auto",
) -> StockTradingEnv:
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "baseline":
        return StockTradingEnv(df=train, **env_kwargs)
    alpha, beta = VARIANT_WEIGHTS[variant]
    config = IntrinsicRewardConfig(
        total_timesteps=total_timesteps,
        alpha=alpha,
        beta=beta,
        warmup_steps=warmup_steps,
        batch_size=min(128, max(2, warmup_steps)),
        replay_capacity=min(
            MAX_INTRINSIC_REPLAY_CAPACITY,
            max(10_000, total_timesteps),
        ),
        seed=seed,
        device=device,
    )
    return IntrinsicRewardStockTradingEnv(
        df=train,
        intrinsic_config=config,
        intrinsic_mode="train",
        **env_kwargs,
    )


def train_variant(
    train: pd.DataFrame,
    variant: str,
    total_timesteps: int,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict,
    ppo_kwargs: dict,
    output_dir: Path | None = None,
    validation: pd.DataFrame | None = None,
    eval_freq: int | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    force_retrain: bool = False,
    show_progress: bool = True,
    training_plan: VariantTrainingPlan | None = None,
) -> VariantRun:
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown variant {variant!r}")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if validation is not None and eval_freq is None:
        raise ValueError("eval_freq is required when validation is enabled")

    set_global_seed(seed)
    plan = training_plan or plan_variant_training(
        train=train,
        validation=validation,
        variant=variant,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        seed=seed,
        warmup_steps=warmup_steps,
        env_kwargs=env_kwargs,
        ppo_kwargs=ppo_kwargs,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_retrain=force_retrain,
    )
    if plan.variant != variant or plan.seed != int(seed):
        raise ValueError("training_plan does not match variant and seed")
    cache_key = plan.cache_key
    cache_payload = plan.cache_payload
    cached_run_dir = plan.cached_run_dir
    environment = build_training_environment(
        train,
        variant,
        total_timesteps,
        seed,
        warmup_steps,
        env_kwargs,
        device=str(ppo_kwargs.get("device", "auto")),
    )
    vector_environment, _ = environment.get_sb_env()

    cache_hit = plan.cache_hit
    if cache_hit:
        model = PPO.load(
            cached_run_dir / "agent_ppo",
            env=vector_environment,
            device=ppo_kwargs.get("device", "auto"),
        )
        if isinstance(environment, IntrinsicRewardStockTradingEnv):
            environment.load_intrinsic_state(cached_run_dir / "intrinsic_reward.pt")
        intrinsic_log = _read_cached_frame(
            cached_run_dir / "intrinsic_training_log.csv"
        )
        validation_log = _read_cached_frame(cached_run_dir / "validation_log.csv")
        with (cached_run_dir / "cache_metadata.json").open(encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
        best_timestep = int(cached_metadata.get("best_timestep", total_timesteps))
        if output_dir is not None:
            run_dir = Path(output_dir) / variant / f"seed_{seed}"
            _write_run_artifacts(
                run_dir,
                model,
                environment,
                intrinsic_log,
                validation_log,
                {**cached_metadata, "loaded_from_cache": True},
            )
        return VariantRun(
            variant=variant,
            seed=seed,
            model=model,
            environment=environment,
            intrinsic_log=intrinsic_log,
            validation_log=validation_log,
            best_timestep=best_timestep,
            cache_hit=True,
            cache_key=cache_key,
        )

    model = PPO(
        "MlpPolicy",
        vector_environment,
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )
    log_every = min(256, max(16, total_timesteps // 200))
    metrics_callback = IntrinsicMetricsCallback(
        log_every=log_every,
        training_frame=train if validation is not None else None,
        validation_frame=validation,
        validation_env_kwargs=env_kwargs if validation is not None else None,
        eval_freq=eval_freq,
    )
    callback: BaseCallback = metrics_callback
    if show_progress:
        callback = CallbackList(
            [
                metrics_callback,
                TqdmTrainingCallback(
                    total_timesteps,
                    f"{variant} | seed {seed}",
                ),
            ]
        )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    intrinsic_log = pd.DataFrame(metrics_callback.records)
    validation_log = pd.DataFrame(metrics_callback.validation_records)
    if metrics_callback.best_policy_state is None:
        best_timestep = total_timesteps
        best_validation_reward = np.nan
    else:
        model.policy.load_state_dict(metrics_callback.best_policy_state)
        best_timestep = metrics_callback.best_timestep
        best_validation_reward = metrics_callback.best_score

    metadata = {
        "schema": TRAIN_CACHE_SCHEMA,
        "complete": True,
        "cache_key": cache_key,
        "best_timestep": best_timestep,
        "best_real_validation_reward": best_validation_reward,
        "payload": cache_payload,
        "loaded_from_cache": False,
    }
    if cached_run_dir is not None:
        _write_run_artifacts(
            cached_run_dir,
            model,
            environment,
            intrinsic_log,
            validation_log,
            metadata,
        )
    if output_dir is not None:
        run_dir = Path(output_dir) / variant / f"seed_{seed}"
        _write_run_artifacts(
            run_dir,
            model,
            environment,
            intrinsic_log,
            validation_log,
            metadata,
        )
    return VariantRun(
        variant=variant,
        seed=seed,
        model=model,
        environment=environment,
        intrinsic_log=intrinsic_log,
        validation_log=validation_log,
        best_timestep=best_timestep,
        cache_hit=False,
        cache_key=cache_key,
    )


def rollout_stock_model(
    frame: pd.DataFrame,
    model: PPO,
    env_kwargs: dict,
) -> dict:
    """Evaluate with the original environment so only financial reward remains."""
    environment = StockTradingEnv(df=frame, **env_kwargs)
    observation, _ = environment.reset()
    states = [np.asarray(observation, dtype=float).copy()]
    terminated = False
    truncated = False
    while not (terminated or truncated):
        previous_day = environment.day
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = environment.step(action)
        if environment.day > previous_day:
            states.append(np.asarray(observation, dtype=float).copy())
    values = np.asarray(environment.asset_memory, dtype=float)
    dates = pd.to_datetime(environment.date_memory)
    state_array = np.asarray(states, dtype=float)
    if len(state_array) != len(values):
        raise RuntimeError("state and account-value histories have different lengths")
    stock_dim = int(env_kwargs["stock_dim"])
    prices = state_array[:, 1 : 1 + stock_dim]
    shares = state_array[:, 1 + stock_dim : 1 + 2 * stock_dim]
    stock_values = prices * shares
    weights = np.column_stack((state_array[:, 0], stock_values)) / values[:, None]
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("portfolio weights do not sum to one")

    actions = np.asarray(environment.actions_memory, dtype=float)
    if len(actions):
        traded_notional = np.abs(actions * prices[:-1]).sum(axis=1)
        turnover = float(np.mean(traded_notional / values[:-1]))
    else:
        turnover = np.nan
    reward_per_day = (
        float(np.mean(environment.rewards_memory) * env_kwargs["reward_scaling"])
        if environment.rewards_memory
        else np.nan
    )
    metrics = performance_metrics_from_values(
        values,
        turnover=turnover,
        reward_per_day=reward_per_day,
    )
    return {
        "values": values,
        "dates": dates,
        "actions": actions,
        "weights": weights,
        "weight_labels": ["Cash", *sorted(frame["tic"].unique())],
        "max_weight": (float(weights[1:].max()) if len(weights) > 1 else np.nan),
        "cost": float(environment.cost),
        "trades": int(environment.trades),
        "metrics": metrics,
    }


def buy_and_hold_rollout(frame: pd.DataFrame, env_kwargs: dict) -> dict:
    """Evaluate an equal-weight, initial-purchase-only Buy & Hold benchmark."""
    prices = (
        frame.pivot_table(index="date", columns="tic", values="close", aggfunc="first")
        .sort_index()
        .astype(float)
    )
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Buy & Hold close prices must be finite and positive")
    tickers = sorted(prices.columns.astype(str))
    prices = prices[tickers]
    initial_amount = float(env_kwargs["initial_amount"])
    costs = np.asarray(env_kwargs["buy_cost_pct"], dtype=float)
    if costs.shape != (len(tickers),):
        raise ValueError("buy_cost_pct must have one value per ticker")

    allocation = initial_amount / len(tickers)
    shares = allocation / (prices.iloc[0].to_numpy(dtype=float) * (1.0 + costs))
    marked_values = prices.to_numpy(dtype=float) @ shares
    values = marked_values.copy()
    values[0] = initial_amount
    dates = pd.to_datetime(prices.index)
    steps = max(len(values) - 1, 1)
    invested_notional = float(np.sum(shares * prices.iloc[0].to_numpy(dtype=float)))
    turnover = invested_notional / initial_amount / steps
    reward_per_day = float(
        np.mean(np.diff(values)) * float(env_kwargs["reward_scaling"])
    )
    metrics = performance_metrics_from_values(
        values,
        turnover=turnover,
        reward_per_day=reward_per_day,
    )
    stock_values = prices.to_numpy(dtype=float) * shares
    weights = stock_values / stock_values.sum(axis=1, keepdims=True)
    return {
        "values": values,
        "dates": dates,
        "actions": np.empty((0, len(tickers)), dtype=float),
        "weights": np.column_stack((np.zeros(len(weights)), weights)),
        "weight_labels": ["Cash", *tickers],
        "cost": float(initial_amount - invested_notional),
        "trades": len(tickers),
        "metrics": metrics,
    }


def evaluate_experiment_periods(
    periods: dict[str, pd.DataFrame],
    runs: Sequence[VariantRun],
    env_kwargs: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict]], pd.DataFrame, dict[str, dict]]:
    """Evaluate all PPO runs and Buy & Hold on train, validation, and test."""
    required_periods = {"train", "validation", "test"}
    missing = sorted(required_periods - set(periods))
    if missing:
        raise ValueError(f"periods is missing {missing}")

    ppo_rows = []
    ppo_results: dict[str, dict[str, dict]] = {}
    benchmark_rows = []
    benchmark_results = {}
    for period in ("train", "validation", "test"):
        frame = periods[period]
        period_results = {}
        for run in runs:
            key = f"{run.variant}__seed_{run.seed}"
            result = rollout_stock_model(frame, run.model, env_kwargs)
            period_results[key] = result
            ppo_rows.append(
                {
                    "period": period,
                    "variant": run.variant,
                    "seed": run.seed,
                    "best_timestep": run.best_timestep,
                    **result["metrics"],
                }
            )
        ppo_results[period] = period_results

        benchmark = buy_and_hold_rollout(frame, env_kwargs)
        benchmark_results[period] = benchmark
        benchmark_rows.append(
            {
                "period": period,
                "variant": "buy_and_hold",
                "seed": pd.NA,
                "best_timestep": np.nan,
                **benchmark["metrics"],
            }
        )
    return (
        pd.DataFrame(ppo_rows),
        ppo_results,
        pd.DataFrame(benchmark_rows),
        benchmark_results,
    )


def evaluate_variant_runs(
    test: pd.DataFrame,
    runs: Sequence[VariantRun],
    env_kwargs: dict,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    results = {
        f"{run.variant}__seed_{run.seed}": rollout_stock_model(
            test, run.model, env_kwargs
        )
        for run in runs
    }
    summary = pd.DataFrame(
        [
            {
                "variant": key.split("__seed_")[0],
                "seed": int(key.split("__seed_")[1]),
                **result["metrics"],
            }
            for key, result in results.items()
        ]
    ).sort_values(["variant", "seed"], ignore_index=True)
    return summary, results


def aggregate_intrinsic_diagnostics(
    intrinsic_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate intrinsic diagnostics across seeds for each reward variant."""
    columns = ("reward_intrinsic", "intrinsic_eta")
    required = {"variant", "seed", "timesteps", *columns}
    missing = sorted(required - set(intrinsic_log.columns))
    if missing:
        raise ValueError(f"intrinsic_log is missing columns {missing}")

    frame = intrinsic_log[list(required)].copy()
    for column in ("seed", "timesteps", *columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_seed_curve = (
        frame.groupby(["variant", "seed", "timesteps"], as_index=False)[list(columns)]
        .mean()
        .sort_values(["variant", "seed", "timesteps"])
    )
    curve = (
        per_seed_curve.groupby(["variant", "timesteps"])[list(columns)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    curve.columns = [
        column if not statistic else f"{column}_{statistic}"
        for column, statistic in curve.columns.to_flat_index()
    ]
    std_columns = [f"{column}_std" for column in columns]
    curve[std_columns] = curve[std_columns].fillna(0.0)

    per_seed_summary = per_seed_curve.groupby(["variant", "seed"], as_index=False)[
        list(columns)
    ].mean()
    summary = (
        per_seed_summary.groupby("variant")[list(columns)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if not statistic else f"{column}_{statistic}"
        for column, statistic in summary.columns.to_flat_index()
    ]
    seed_counts = (
        per_seed_summary.groupby("variant")["seed"].nunique().rename("seed_count")
    )
    summary = summary.merge(seed_counts, on="variant", validate="one_to_one")
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return curve, summary.sort_values("variant", ignore_index=True)


def aggregate_evaluation_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation of test metrics across seeds."""
    required = {"variant", "seed", *EVALUATION_METRICS}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing columns {missing}")
    if summary.empty:
        return pd.DataFrame()

    frame = summary[["variant", "seed", *EVALUATION_METRICS]].copy()
    for column in ("seed", *EVALUATION_METRICS):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    grouped = (
        frame.groupby("variant")[list(EVALUATION_METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        column if not statistic else f"{column}_{statistic}"
        for column, statistic in grouped.columns.to_flat_index()
    ]
    seed_counts = frame.groupby("variant")["seed"].nunique().rename("seed_count")
    grouped = grouped.merge(seed_counts, on="variant", validate="one_to_one")
    std_columns = [f"{metric}_std" for metric in EVALUATION_METRICS]
    grouped[std_columns] = grouped[std_columns].fillna(0.0)
    return grouped.sort_values("variant", ignore_index=True)


def aggregate_equity_curves(
    evaluation_results: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-seed rollouts to long form and aggregate by variant/date."""
    frames = []
    for key, result in evaluation_results.items():
        if "__seed_" not in key:
            raise ValueError(f"evaluation key must contain '__seed_': {key!r}")
        variant, seed_text = key.rsplit("__seed_", maxsplit=1)
        dates = pd.to_datetime(result["dates"])
        values = np.asarray(result["values"], dtype=float)
        if len(dates) != len(values):
            raise ValueError(f"{key}: dates and values have different lengths")
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: account values contain NaN or infinity")
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "account_value": values,
                    "variant": variant,
                    "seed": int(seed_text),
                    "run": key,
                }
            )
        )
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    long_frame = pd.concat(frames, ignore_index=True)
    if long_frame.duplicated(["variant", "seed", "date"]).any():
        raise ValueError("evaluation results contain duplicate variant/seed/date rows")
    aggregated = (
        long_frame.groupby(["variant", "date"])["account_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "seed_count"})
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return long_frame, aggregated


def _variant_order(values: Sequence[str]) -> list[str]:
    available = set(values)
    preferred = (
        "baseline",
        "paper_surprise",
        "robust_surprise",
        "dejavu",
        "paper_surprise_dejavu",
        "robust_surprise_dejavu",
        "combined",
    )
    known = [variant for variant in preferred if variant in available]
    return known + sorted(available - set(known))


def _mean_std_text(
    row: pd.Series,
    metric: str,
    *,
    percent: bool = False,
    decimals: int = 3,
) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if not np.isfinite(mean):
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{mean * scale:,.{decimals}f}{suffix} ± "
        f"{std * scale:,.{decimals}f}{suffix}"
    )


def plot_intrinsic_diagnostics(
    intrinsic_log: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Plot reward-mechanism diagnostics separately from common PPO curves."""
    curve, summary = aggregate_intrinsic_diagnostics(intrinsic_log)
    if curve.empty:
        raise ValueError("intrinsic_log must contain at least one observation")

    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(3.0, 3.0, 1.15))
    axes = [
        figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)
    ]
    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")

    variants = _variant_order(curve["variant"].unique())
    palette = (
        "#0057B8",
        "#E66100",
        "#00856A",
        "#C51B7D",
        "#6A3D9A",
        "#8C6D1F",
    )
    colors = {
        variant: palette[index % len(palette)] for index, variant in enumerate(variants)
    }
    panels = (
        (
            axes[0],
            (
                ("reward_ext", "external reward", "-"),
                ("reward_intrinsic", "intrinsic reward", "--"),
            ),
            "External and intrinsic reward",
        ),
        (
            axes[1],
            (
                ("reward_surprise", "surprise bonus", "-"),
                ("reward_dejavu", "déjà vu bonus", "--"),
            ),
            "Scaled intrinsic bonuses",
        ),
        (
            axes[2],
            (
                ("reward_surprise_raw", "raw surprise", "-"),
                ("reward_dejavu_raw", "raw déjà vu", "--"),
            ),
            "Raw novelty scores",
        ),
        (
            axes[3],
            (
                ("intrinsic_eta", "eta", "-"),
                ("intrinsic_time_decay", "time decay", "--"),
                (
                    "intrinsic_volatility_factor",
                    "volatility factor",
                    ":",
                ),
            ),
            "Dynamic intrinsic weighting",
        ),
        (
            axes[4],
            (
                ("intrinsic_surprise_loss", "surprise loss", "-"),
                ("intrinsic_dejavu_loss", "déjà vu loss", "--"),
            ),
            "Novelty-model losses",
        ),
        (
            axes[5],
            (
                ("reward_train", "sampled training reward", "-"),
                ("reward_ext", "external component", "--"),
            ),
            "Training reward decomposition",
        ),
    )
    for axis, specs, title in panels:
        plotted = False
        for variant in variants:
            variant_frame = intrinsic_log[intrinsic_log["variant"] == variant]
            for metric, metric_label, linestyle in specs:
                label = f"{variant} — {metric_label}"
                plotted |= _plot_aggregate_band(
                    axis,
                    variant_frame,
                    metric,
                    label,
                    colors[variant],
                    linestyle=linestyle,
                )
        axis.set_title(title)
        axis.set_xlabel("training timesteps")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=7, ncol=1)
        else:
            axis.text(
                0.5,
                0.5,
                "No observations",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6B7280",
            )

    table_rows = []
    for variant in variants:
        row = summary.loc[summary["variant"] == variant].iloc[0]
        reward_text = (
            f"{float(row['reward_intrinsic_mean']):.2e} ± "
            f"{float(row['reward_intrinsic_std']):.2e}"
        )
        table_rows.append(
            [
                variant,
                str(int(row["seed_count"])),
                reward_text,
                _mean_std_text(row, "intrinsic_eta", decimals=5),
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Model", "Seeds", "Mean intrinsic reward ± SD", "Mean eta ± SD"),
        cellLoc="center",
        loc="center",
    )
    _style_report_table(table, font_size=9)
    table.scale(1.0, 1.35)
    figure.suptitle(
        "Intrinsic-reward diagnostics: across-seed mean ± 1 SD",
        fontsize=17,
    )
    return figure, curve, summary


def plot_evaluation_summary(
    summary: pd.DataFrame,
    evaluation_results: dict[str, dict],
    title: str = "Out-of-sample account value",
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Plot model-level equity curves and a cross-seed metric table."""
    long_frame, curve = aggregate_equity_curves(evaluation_results)
    metric_summary = aggregate_evaluation_metrics(summary)
    if curve.empty or metric_summary.empty:
        raise ValueError("evaluation results and summary must not be empty")

    figure = plt.figure(figsize=(13, 8.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 1.6))
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])
    table_axis.axis("off")

    variants = _variant_order(curve["variant"].unique())
    colors = dict(zip(variants, plt.get_cmap("tab10").colors))
    for variant in variants:
        frame = curve[curve["variant"] == variant]
        dates = pd.to_datetime(frame["date"])
        mean = frame["mean"].to_numpy(dtype=float)
        std = frame["std"].to_numpy(dtype=float)
        seed_count = int(frame["seed_count"].max())
        axis.plot(
            dates,
            mean,
            color=colors[variant],
            label=f"{variant} (n={seed_count})",
        )
        axis.fill_between(
            dates,
            mean - std,
            mean + std,
            color=colors[variant],
            alpha=0.16,
            linewidth=0,
        )
    axis.set_title(title)
    axis.set_xlabel("date")
    axis.set_ylabel("account value")
    axis.grid(alpha=0.25)
    axis.legend(title="model", fontsize=8)

    table_rows = []
    for variant in variants:
        row = metric_summary.loc[metric_summary["variant"] == variant].iloc[0]
        table_rows.append(
            [
                variant,
                str(int(row["seed_count"])),
                _mean_std_text(row, "final_value", decimals=0),
                _mean_std_text(row, "cumulative_return", percent=True, decimals=2),
                _mean_std_text(row, "cagr", percent=True, decimals=2),
                _mean_std_text(row, "sharpe", decimals=3),
                _mean_std_text(row, "annualized_volatility", percent=True, decimals=2),
                _mean_std_text(row, "max_drawdown", percent=True, decimals=2),
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "Model",
            "Seeds",
            "Final value",
            "Return",
            "CAGR",
            "Sharpe",
            "Volatility",
            "Max DD",
        ),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    figure.suptitle("Model comparison: across-seed mean ± 1 SD")
    return figure, long_frame, curve, metric_summary


def _style_report_table(table, font_size: float = 8.5) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row_number, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row_number == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row_number % 2 == 0:
            cell.set_facecolor("#F3F4F6")


def _aggregate_seed_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    usable = frame.dropna(subset=[metric])
    if usable.empty:
        return pd.DataFrame()
    per_seed = usable.groupby(["seed", "timesteps"], as_index=False)[metric].mean()
    aggregated = (
        per_seed.groupby("timesteps")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated


def _plot_aggregate_band(
    axis,
    frame: pd.DataFrame,
    metric: str,
    label: str,
    color: str,
    linestyle: str = "-",
) -> bool:
    if frame.empty or metric not in frame:
        return False
    aggregated = _aggregate_seed_metric(frame, metric)
    if aggregated.empty:
        return False
    timesteps = aggregated["timesteps"].to_numpy(dtype=float)
    mean = aggregated["mean"].to_numpy(dtype=float)
    std = aggregated["std"].to_numpy(dtype=float)
    axis.plot(
        timesteps,
        mean,
        color=color,
        linestyle=linestyle,
        marker="o" if len(timesteps) <= 12 else None,
        label=label,
    )
    axis.fill_between(
        timesteps,
        mean - std,
        mean + std,
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    return True


def plot_learning_curves_by_variant(
    runs: Sequence[VariantRun],
) -> dict[str, plt.Figure]:
    """Create the common original-experiment learning panels per variant."""
    figures = {}
    for variant in _variant_order([run.variant for run in runs]):
        variant_runs = [run for run in runs if run.variant == variant]
        log_frames = []
        validation_frames = []
        for run in variant_runs:
            if not run.intrinsic_log.empty:
                frame = run.intrinsic_log.copy()
                frame["seed"] = run.seed
                log_frames.append(frame)
            if not run.validation_log.empty:
                frame = run.validation_log.copy()
                frame["seed"] = run.seed
                validation_frames.append(frame)
        training_log = (
            pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
        )
        validation_log = (
            pd.concat(validation_frames, ignore_index=True)
            if validation_frames
            else pd.DataFrame()
        )
        if not training_log.empty:
            training_log = training_log.sort_values(["seed", "timesteps"])
            training_log["sampled_reward_rolling"] = training_log.groupby(
                "seed", sort=False
            )["reward_train"].transform(
                lambda values: values.rolling(10, min_periods=1).mean()
            )
        if not validation_log.empty:
            validation_log["generalization_gap"] = (
                validation_log["train_reward_mean"]
                - validation_log["real_valid_reward"]
            )

        figure = plt.figure(figsize=(18, 11), constrained_layout=True)
        grid = figure.add_gridspec(3, 3, height_ratios=(3.0, 3.0, 1.15))
        axes = [
            figure.add_subplot(grid[row, column])
            for row in range(2)
            for column in range(3)
        ]
        table_axis = figure.add_subplot(grid[2, :])
        table_axis.axis("off")

        _plot_aggregate_band(
            axes[0],
            training_log,
            "sampled_reward_rolling",
            "rolling mean (10)",
            "#0057B8",
        )
        axes[0].axhline(0, color="#6B7280", linewidth=0.8)
        axes[0].set_title("Sampled-policy reward: rolling mean (10)")

        _plot_aggregate_band(
            axes[1],
            validation_log,
            "train_reward_mean",
            "training source",
            "#0057B8",
        )
        _plot_aggregate_band(
            axes[1],
            validation_log,
            "real_valid_reward",
            "real validation",
            "#E66100",
            linestyle="--",
        )
        axes[1].set_title("Deterministic financial reward per day")

        _plot_aggregate_band(
            axes[2],
            validation_log,
            "generalization_gap",
            "train − real validation",
            "#C51B7D",
        )
        axes[2].axhline(0, color="#6B7280", linestyle="--", linewidth=0.8)
        axes[2].set_title("PPO generalization gap")

        _plot_aggregate_band(
            axes[3],
            validation_log,
            "real_valid_reward",
            "real validation",
            "#E66100",
            linestyle="--",
        )
        axes[3].set_title("Real validation financial reward")

        behaviour_twin = axes[4].twinx()
        _plot_aggregate_band(
            axes[4],
            validation_log,
            "real_valid_turnover",
            "turnover",
            "#00856A",
        )
        _plot_aggregate_band(
            behaviour_twin,
            validation_log,
            "real_valid_max_weight",
            "max weight",
            "#6A3D9A",
            linestyle="--",
        )
        axes[4].set_title("Behaviour on real validation")
        axes[4].set_ylabel("mean daily turnover")
        behaviour_twin.set_ylabel("max portfolio weight")

        internals_twin = axes[5].twinx()
        _plot_aggregate_band(
            axes[5],
            training_log,
            "explained_variance",
            "explained variance",
            "#0057B8",
        )
        _plot_aggregate_band(
            axes[5],
            training_log,
            "value_loss",
            "value loss",
            "#E66100",
            linestyle="--",
        )
        _plot_aggregate_band(
            internals_twin,
            training_log,
            "entropy_loss",
            "entropy loss",
            "#C51B7D",
            linestyle=":",
        )
        axes[5].set_title("PPO internals")
        axes[5].set_ylabel("explained variance / value loss")
        internals_twin.set_ylabel("entropy loss")

        best_steps = np.asarray(
            [run.best_timestep for run in variant_runs], dtype=float
        )
        finite_best_steps = best_steps[np.isfinite(best_steps)]
        if len(finite_best_steps):
            mean_best_step = float(finite_best_steps.mean())
            for axis in axes[1:5]:
                axis.axvline(
                    mean_best_step,
                    color="#708238",
                    linestyle=":",
                    alpha=0.75,
                    label="mean best step",
                )

        for axis in axes:
            axis.set_xlabel("timesteps")
            axis.grid(alpha=0.2, color="#D1D5DB")
            handles, labels = axis.get_legend_handles_labels()
            if axis is axes[4]:
                twin_handles, twin_labels = behaviour_twin.get_legend_handles_labels()
                handles += twin_handles
                labels += twin_labels
            if axis is axes[5]:
                twin_handles, twin_labels = internals_twin.get_legend_handles_labels()
                handles += twin_handles
                labels += twin_labels
            if handles:
                axis.legend(handles, labels, fontsize=8)

        final_validation = []
        best_validation = []
        for run in variant_runs:
            if not run.validation_log.empty:
                final_validation.append(
                    float(run.validation_log.iloc[-1]["real_valid_reward"])
                )
                best_validation.append(
                    float(run.validation_log["real_valid_reward"].max())
                )
        final_rewards = []
        for run in variant_runs:
            if not run.intrinsic_log.empty:
                final_rewards.append(float(run.intrinsic_log.iloc[-1]["reward_train"]))

        def pair_text(values: Sequence[float], decimals: int = 3) -> str:
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            if not len(array):
                return "—"
            std = array.std(ddof=1) if len(array) > 1 else 0.0
            return f"{array.mean():,.{decimals}f} ± {std:,.{decimals}f}"

        table = table_axis.table(
            cellText=[
                [
                    variant,
                    str(len(variant_runs)),
                    pair_text(best_steps, decimals=0),
                    pair_text(final_rewards, decimals=5),
                    pair_text(best_validation, decimals=5),
                    pair_text(final_validation, decimals=5),
                ]
            ],
            colLabels=(
                "Model",
                "Seeds",
                "Best validation step ± SD",
                "Final training reward ± SD",
                "Best validation reward ± SD",
                "Final validation reward ± SD",
            ),
            cellLoc="center",
            loc="center",
        )
        _style_report_table(table, font_size=9)
        table.scale(1.0, 1.35)
        figure.suptitle(
            f"{variant}: PPO learning curves across seeds (mean ± 1 SD)",
            fontsize=17,
        )
        figures[variant] = figure
    return figures


def select_median_sharpe_runs(test_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the seed closest to each variant's median test Sharpe."""
    required = {"variant", "seed", "sharpe"}
    missing = sorted(required - set(test_summary.columns))
    if missing:
        raise ValueError(f"test_summary is missing columns {missing}")
    rows = []
    for variant, frame in test_summary.groupby("variant", sort=False):
        candidates = frame.dropna(subset=["sharpe"]).copy()
        if candidates.empty:
            selected = frame.sort_values("seed").iloc[len(frame) // 2].copy()
            median = np.nan
            distance = np.nan
        else:
            median = float(candidates["sharpe"].median())
            candidates["_distance"] = (candidates["sharpe"] - median).abs()
            selected = candidates.sort_values(["_distance", "seed"]).iloc[0].copy()
            distance = float(selected["_distance"])
        selected["median_test_sharpe"] = median
        selected["sharpe_distance_to_median"] = distance
        rows.append(selected.drop(labels=["_distance"], errors="ignore"))
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_sharpe_seed_selection(
    test_summary: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show every test seed and mark the median-Sharpe representative."""
    representatives = select_median_sharpe_runs(test_summary)
    variants = _variant_order(test_summary["variant"].unique())
    colors = dict(zip(variants, plt.get_cmap("tab10").colors))
    figure = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.5, 1.25))
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])
    table_axis.axis("off")

    table_rows = []
    for position, variant in enumerate(variants):
        rows = test_summary[test_summary["variant"] == variant].sort_values("seed")
        offsets = np.linspace(-0.13, 0.13, max(len(rows), 1))
        for offset, (_, row) in zip(offsets, rows.iterrows()):
            if pd.isna(row["sharpe"]):
                continue
            axis.scatter(
                row["sharpe"],
                position + offset,
                color=colors[variant],
                alpha=0.5,
            )
            axis.annotate(
                f"s{int(row['seed'])}",
                (row["sharpe"], position + offset),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        selected = representatives[representatives["variant"] == variant].iloc[0]
        axis.scatter(
            selected["sharpe"],
            position,
            marker="D",
            s=110,
            color=colors[variant],
            edgecolor="#111827",
            zorder=4,
        )
        table_rows.append(
            [
                variant,
                str(len(rows)),
                f"{selected['median_test_sharpe']:.3f}",
                f"seed {int(selected['seed'])}",
                f"{selected['sharpe']:.3f}",
            ]
        )
    axis.axvline(0, color="#6B7280", linewidth=0.8)
    axis.set_yticks(range(len(variants)), variants)
    axis.invert_yaxis()
    axis.set_xlabel("Test Sharpe ratio")
    axis.set_title("Test Sharpe across training seeds")
    axis.grid(axis="x", alpha=0.2, color="#D1D5DB")
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "Model",
            "Seeds",
            "Median Sharpe",
            "Selected seed",
            "Selected Sharpe",
        ),
        cellLoc="center",
        loc="center",
    )
    _style_report_table(table)
    table.scale(1.0, 1.35)
    figure.suptitle("Diamonds identify the seed closest to each model median")
    return figure, representatives


def plot_test_metric_comparison(
    test_summary: pd.DataFrame,
    buy_and_hold_metrics: pd.Series,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Compare cross-seed PPO test metrics against deterministic Buy & Hold."""
    metrics = (
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "turnover",
    )
    required = {"variant", "seed", *metrics}
    missing = sorted(required - set(test_summary.columns))
    if missing:
        raise ValueError(f"test_summary is missing columns {missing}")
    grouped = (
        test_summary.groupby("variant")[list(metrics)]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        column if not stat else f"{column}_{stat}"
        for column, stat in grouped.columns.to_flat_index()
    ]
    grouped[[f"{metric}_std" for metric in metrics]] = grouped[
        [f"{metric}_std" for metric in metrics]
    ].fillna(0.0)
    counts = test_summary.groupby("variant")["seed"].nunique().rename("seed_count")
    grouped = grouped.merge(counts, on="variant", validate="one_to_one")

    variants = _variant_order(grouped["variant"].unique())
    order = [*variants, "buy_and_hold"]
    colors = dict(zip(variants, plt.get_cmap("tab10").colors))
    colors["buy_and_hold"] = "#111111"
    figure = plt.figure(figsize=(18, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(3.5, 1.55))
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    plot_specs = (
        ("cumulative_return", "Cumulative return", True),
        ("sharpe", "Sharpe ratio", False),
        ("max_drawdown", "Maximum drawdown", True),
        ("turnover", "Mean daily turnover", True),
    )
    y_positions = np.arange(len(order))
    for axis, (metric, title, percent) in zip(axes, plot_specs):
        for position, variant in enumerate(variants):
            row = grouped[grouped["variant"] == variant].iloc[0]
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            axis.errorbar(
                mean,
                position,
                xerr=std,
                fmt="o",
                color=colors[variant],
                capsize=4,
            )
        benchmark_value = float(buy_and_hold_metrics[metric])
        axis.scatter(
            benchmark_value,
            len(order) - 1,
            marker="D",
            color=colors["buy_and_hold"],
        )
        axis.axvline(0, color="#6B7280", linewidth=0.8)
        axis.set_title(title)
        axis.set_yticks(y_positions)
        display_order = [*variants, "Buy & Hold"]
        axis.set_yticklabels(display_order if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2, color="#D1D5DB")
        if percent:
            axis.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")

    table_rows = []
    for variant in variants:
        row = grouped[grouped["variant"] == variant].iloc[0]
        table_rows.append(
            [
                variant,
                str(int(row["seed_count"])),
                _mean_std_text(row, "cumulative_return", percent=True, decimals=2),
                _mean_std_text(row, "sharpe", decimals=3),
                _mean_std_text(row, "max_drawdown", percent=True, decimals=2),
                _mean_std_text(row, "turnover", percent=True, decimals=2),
            ]
        )
    table_rows.append(
        [
            "Buy & Hold",
            "—",
            f"{float(buy_and_hold_metrics['cumulative_return']):.2%}",
            f"{float(buy_and_hold_metrics['sharpe']):.3f}",
            f"{float(buy_and_hold_metrics['max_drawdown']):.2%}",
            f"{float(buy_and_hold_metrics['turnover']):.2%}",
        ]
    )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Model", "Seeds", "Return", "Sharpe", "Max DD", "Turnover"),
        cellLoc="center",
        loc="center",
    )
    _style_report_table(table)
    table.scale(1.0, 1.35)
    figure.suptitle("External-reward test metrics: model mean ± 1 SD vs Buy & Hold")
    return figure, grouped.sort_values("variant", ignore_index=True)


def plot_period_equity_comparison(
    period: str,
    evaluation_results: dict[str, dict],
    benchmark_result: dict,
    title: str,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Plot across-seed Growth of $1 curves and deterministic Buy & Hold."""
    normalized_results = {}
    for key, result in evaluation_results.items():
        values = np.asarray(result["values"], dtype=float)
        normalized_results[key] = {
            "dates": result["dates"],
            "values": values / values[0],
        }
    long_frame, curve = aggregate_equity_curves(normalized_results)
    variants = _variant_order(curve["variant"].unique())
    colors = dict(zip(variants, plt.get_cmap("tab10").colors))
    figure = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 1.5))
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])
    table_axis.axis("off")

    ranking_rows = []
    for variant in variants:
        frame = curve[curve["variant"] == variant]
        dates = pd.to_datetime(frame["date"])
        mean = frame["mean"].to_numpy(dtype=float)
        std = frame["std"].to_numpy(dtype=float)
        seed_count = int(frame["seed_count"].max())
        axis.plot(
            dates,
            mean,
            color=colors[variant],
            label=f"{variant} (n={seed_count})",
        )
        axis.fill_between(
            dates,
            mean - std,
            mean + std,
            color=colors[variant],
            alpha=0.16,
            linewidth=0,
        )
        ending = (
            long_frame[long_frame["variant"] == variant]
            .groupby("seed")["account_value"]
            .last()
        )
        ranking_rows.append(
            {
                "model": variant,
                "seeds": seed_count,
                "ending_mean": float(ending.mean()),
                "ending_std": float(ending.std(ddof=1)) if len(ending) > 1 else 0.0,
            }
        )
    benchmark_values = np.asarray(benchmark_result["values"], dtype=float)
    benchmark_growth = benchmark_values / benchmark_values[0]
    axis.plot(
        pd.to_datetime(benchmark_result["dates"]),
        benchmark_growth,
        color="#111111",
        linestyle="--",
        linewidth=2.2,
        label="Buy & Hold",
    )
    ranking_rows.append(
        {
            "model": "Buy & Hold",
            "seeds": 0,
            "ending_mean": float(benchmark_growth[-1]),
            "ending_std": 0.0,
        }
    )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["ending_mean", "model"], ascending=[False, True], ignore_index=True
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    axis.axhline(1.0, color="#6B7280", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel(f"{period} date")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.2, color="#D1D5DB")
    axis.legend(fontsize=9, ncol=2)

    table_rows = [
        [
            int(row["rank"]),
            row["model"],
            "—" if row["seeds"] == 0 else int(row["seeds"]),
            f"{row['ending_mean']:.3f} ± {row['ending_std']:.3f}",
            f"{row['ending_mean'] - 1.0:.2%} ± {row['ending_std']:.2%}",
        ]
        for _, row in ranking.iterrows()
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Rank", "Model", "Seeds", "End $1 ± SD", "Return ± SD"),
        cellLoc="center",
        loc="center",
    )
    _style_report_table(table)
    table.scale(1.0, 1.35)
    figure.suptitle("PPO model means vs Buy & Hold")
    return figure, long_frame, ranking


def plot_test_portfolio_weights(
    evaluation_results: dict[str, dict],
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot mean test portfolio weights across seeds for every PPO variant."""
    frames = []
    labels: list[str] | None = None
    for key, result in evaluation_results.items():
        if "__seed_" not in key:
            raise ValueError(f"evaluation key must contain '__seed_': {key!r}")
        variant, seed_text = key.rsplit("__seed_", maxsplit=1)
        result_labels = list(result["weight_labels"])
        if labels is None:
            labels = result_labels
        elif labels != result_labels:
            raise ValueError("asset order differs across evaluation runs")
        weights = np.asarray(result["weights"], dtype=float)
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"portfolio weights do not sum to one for {key}")
        frame = pd.DataFrame(weights, columns=result_labels)
        frame["date"] = pd.to_datetime(result["dates"])
        frame["variant"] = variant
        frame["seed"] = int(seed_text)
        frames.append(frame)
    if not frames or labels is None:
        raise ValueError("evaluation_results must not be empty")

    weights_long = pd.concat(frames, ignore_index=True)
    mean_weights = weights_long.groupby(["variant", "date"], as_index=False)[
        labels
    ].mean()
    variants = _variant_order(mean_weights["variant"].unique())
    colors = ["#D1D5DB", "#0057B8", "#E66100", "#00856A", "#C51B7D", "#6A3D9A"]
    figure = plt.figure(
        figsize=(15, max(10, 2.7 * len(variants) + 2.5)),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        len(variants) + 2,
        1,
        height_ratios=(0.55, *([3.0] * len(variants)), 1.5),
    )
    legend_axis = figure.add_subplot(grid[0, 0])
    legend_axis.axis("off")
    axes = [figure.add_subplot(grid[index + 1, 0]) for index in range(len(variants))]
    table_axis = figure.add_subplot(grid[-1, 0])
    table_axis.axis("off")
    table_rows = []
    for axis, variant in zip(axes, variants):
        frame = mean_weights[mean_weights["variant"] == variant]
        axis.stackplot(
            pd.to_datetime(frame["date"]),
            frame[labels].to_numpy(dtype=float).T,
            labels=labels,
            colors=colors[: len(labels)],
            alpha=0.85,
        )
        seed_count = weights_long.loc[
            weights_long["variant"] == variant, "seed"
        ].nunique()
        axis.set_title(f"{variant} | mean allocation across {seed_count} seeds")
        axis.set_ylabel("Weight")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
        averages = weights_long[weights_long["variant"] == variant][labels].mean()
        table_rows.append(
            [variant, str(seed_count), *[f"{averages[label]:.1%}" for label in labels]]
        )
    axes[-1].set_xlabel("test date")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        legend_labels,
        title="Asset",
        loc="center",
        ncol=len(labels),
    )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Model", "Seeds", *labels),
        cellLoc="center",
        loc="center",
    )
    _style_report_table(table, font_size=8)
    table.scale(1.0, 1.35)
    figure.suptitle("Actual test portfolio weights: across-seed mean")
    return figure, mean_weights
