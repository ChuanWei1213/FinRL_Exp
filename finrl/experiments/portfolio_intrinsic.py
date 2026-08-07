from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from finrl.experiments.notebook_utils import EIIEExtractor
from finrl.experiments.notebook_utils import EpisodeLogger
from finrl.experiments.notebook_utils import performance_metrics_from_values
from finrl.experiments.notebook_utils import PriceScaleByTicker
from finrl.experiments.notebook_utils import set_global_seed
from finrl.experiments.notebook_utils import slice_evaluation_with_lookback
from finrl.experiments.notebook_utils import slice_period
from finrl.experiments.notebook_utils import TqdmTrainingCallback
from finrl.experiments.notebook_utils import validate_periods
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
    PortfolioOptimizationGymnasiumEnv,
)
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_intrinsic import (
    IntrinsicRewardPortfolioOptimizationEnv,
)
from finrl.meta.rewards import IntrinsicRewardConfig
from finrl.meta.rewards import IntrinsicRewardController
from finrl.meta.rewards import PaperFaithfulIntrinsicRewardController
from finrl.meta.rewards import RobustIntrinsicRewardController


PRICE_COLUMNS = ("close", "high", "low")
VARIANT_SPECS = {
    "baseline": {
        "alpha": 0.0,
        "beta": 0.0,
        "surprise_mode": "none",
    },
    "paper_surprise": {
        "alpha": 0.05,
        "beta": 0.0,
        "surprise_mode": "paper_sum_nll",
        "dimension_adjust_alpha": True,
    },
    "robust_surprise": {
        "alpha": 0.05,
        "beta": 0.0,
        "surprise_mode": "robust_centered_z",
    },
    "dejavu": {
        "alpha": 0.0,
        "beta": 0.05,
        "surprise_mode": "none",
    },
    "paper_surprise_dejavu": {
        "alpha": 0.05,
        "beta": 0.05,
        "surprise_mode": "paper_sum_nll",
        "dimension_adjust_alpha": True,
    },
    "robust_surprise_dejavu": {
        "alpha": 0.05,
        "beta": 0.05,
        "surprise_mode": "robust_centered_z",
    },
}
VARIANT_WEIGHTS = {
    variant: (float(spec["alpha"]), float(spec["beta"]))
    for variant, spec in VARIANT_SPECS.items()
}
SURPRISE_CONTROLLER_CLASSES = {
    "none": IntrinsicRewardController,
    "paper_sum_nll": PaperFaithfulIntrinsicRewardController,
    "robust_centered_z": RobustIntrinsicRewardController,
}
TRAIN_CACHE_SCHEMA = "portfolio_intrinsic_eiie_v2"
MAX_INTRINSIC_REPLAY_CAPACITY = 100_000


@dataclass
class PortfolioVariantRun:
    variant: str
    commission_name: str
    commission: float
    seed: int
    model: PPO
    environment: PortfolioOptimizationGymnasiumEnv | None
    intrinsic_log: pd.DataFrame
    validation_log: pd.DataFrame
    best_timestep: int
    run_dir: Path
    cache_hit: bool = False
    cache_key: str = ""


@dataclass(frozen=True)
class PortfolioTrainingPlan:
    variant: str
    commission_name: str
    commission: float
    seed: int
    cache_key: str
    cache_payload: dict[str, Any]
    run_dir: Path
    cache_hit: bool


def prepare_real_portfolio_periods(
    real_ohlcv: pd.DataFrame,
    expected_tickers: Sequence[str],
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    *,
    time_window: int = 50,
    price_columns: Sequence[str] = PRICE_COLUMNS,
) -> tuple[dict[str, pd.DataFrame], PriceScaleByTicker]:
    """Build the same scaled real-data splits as the Synthetic-vs-Real control."""
    validate_periods(
        (
            ("train", train_start, train_end),
            ("validation", validation_start, validation_end),
            ("test", test_start, test_end),
        )
    )
    if time_window <= 1:
        raise ValueError("time_window must be greater than one")
    required = {"date", "tic", *price_columns}
    missing = sorted(required - set(real_ohlcv.columns))
    if missing:
        raise ValueError(f"real_ohlcv is missing {missing}")

    frame = real_ohlcv[["date", "tic", *price_columns]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    expected = sorted(str(ticker).upper() for ticker in expected_tickers)
    actual = sorted(frame["tic"].astype(str).str.upper().unique())
    if actual != expected:
        raise ValueError(f"expected tickers {expected}, got {actual}")

    train_raw = slice_period(
        frame,
        train_start,
        train_end,
        "real train",
        minimum_dates=time_window + 1,
    )
    validation_raw = slice_evaluation_with_lookback(
        frame,
        validation_start,
        validation_end,
        "real validation",
        time_window=time_window,
    )
    test_raw = slice_evaluation_with_lookback(
        frame,
        test_start,
        test_end,
        "real test",
        time_window=time_window,
    )
    scaler = PriceScaleByTicker(expected, price_columns).fit([train_raw])
    periods = {
        "train": scaler.transform(train_raw),
        "validation": scaler.transform(validation_raw),
        "test": scaler.transform(test_raw),
    }
    return periods, scaler


def portfolio_environment_kwargs(
    commission: float,
    *,
    cwd: str | Path,
    initial_amount: int = 100_000,
    time_window: int = 50,
    reward_scaling: float = 100.0,
    price_columns: Sequence[str] = PRICE_COLUMNS,
) -> dict[str, Any]:
    """Return the exact environment settings used by the real-data control."""
    if commission < 0:
        raise ValueError("commission must be non-negative")
    return {
        "initial_amount": initial_amount,
        "time_window": time_window,
        "features": list(price_columns),
        "normalize_df": None,
        "reward_scaling": reward_scaling,
        "action_space_mode": "symmetric",
        "action_scale": 5.0,
        "return_last_action": True,
        "plot_on_terminal": False,
        "cwd": str(cwd),
        "comission_fee_pct": float(commission),
    }


def portfolio_ppo_kwargs(device: str = "cpu") -> dict[str, Any]:
    """Return the exact PPO and EIIE settings used by the real-data control."""
    return {
        "learning_rate": 3e-4,
        "n_steps": 2_048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "device": device,
        "verbose": 0,
        "policy_kwargs": {
            "log_std_init": -2.0,
            "features_extractor_class": EIIEExtractor,
            "features_extractor_kwargs": {
                "k_size": 3,
                "conv_mid": 2,
                "conv_final": 20,
            },
        },
    }


def resolve_variant_reward_config(
    train: pd.DataFrame,
    variant: str,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Resolve nominal weights to the effective controller configuration."""
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unknown variant {variant!r}")
    spec = VARIANT_SPECS[variant]
    stock_dimension = int(train["tic"].nunique())
    observation_dim = (
        len(env_kwargs["features"]) * stock_dimension * int(env_kwargs["time_window"])
        + stock_dimension
        + 1
    )
    nominal_alpha = float(spec["alpha"])
    effective_alpha = nominal_alpha
    if spec.get("dimension_adjust_alpha", False):
        effective_alpha = nominal_alpha / observation_dim
    surprise_mode = str(spec["surprise_mode"])
    return {
        "nominal_alpha": nominal_alpha,
        "effective_alpha": effective_alpha,
        "beta": float(spec["beta"]),
        "surprise_mode": surprise_mode,
        "controller_class": SURPRISE_CONTROLLER_CLASSES[surprise_mode],
        "observation_dim": observation_dim,
        "dimension_adjust_alpha": bool(spec.get("dimension_adjust_alpha", False)),
    }


def build_portfolio_training_environment(
    train: pd.DataFrame,
    variant: str,
    total_timesteps: int,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict[str, Any],
    *,
    device: str = "auto",
    replay_capacity: int = MAX_INTRINSIC_REPLAY_CAPACITY,
) -> PortfolioOptimizationGymnasiumEnv:
    if variant == "baseline":
        return PortfolioOptimizationGymnasiumEnv(df=train, **env_kwargs)
    reward_config = resolve_variant_reward_config(train, variant, env_kwargs)
    config = IntrinsicRewardConfig(
        total_timesteps=total_timesteps,
        alpha=reward_config["effective_alpha"],
        beta=reward_config["beta"],
        warmup_steps=warmup_steps,
        batch_size=min(128, max(2, warmup_steps)),
        replay_capacity=min(replay_capacity, max(10_000, total_timesteps)),
        seed=seed,
        device=device,
    )
    return IntrinsicRewardPortfolioOptimizationEnv(
        df=train,
        intrinsic_config=config,
        intrinsic_mode="train",
        intrinsic_controller_class=reward_config["controller_class"],
        **env_kwargs,
    )


def rollout_portfolio_model(
    frame: pd.DataFrame,
    model: PPO | None,
    env_kwargs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run deterministic external-reward evaluation in the original environment."""
    environment = PortfolioOptimizationGymnasiumEnv(frame, **env_kwargs)
    observation, _ = environment.reset(seed=seed)
    n_actions = environment.action_space.shape[0]
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if model is None:
            action = np.concatenate([[-1.0], np.zeros(n_actions - 1)]).astype(
                np.float32
            )
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = environment.step(action)

    values = np.asarray(environment._asset_memory["final"], dtype=float)
    dates = pd.to_datetime(environment._date_memory)
    chosen = np.asarray(environment._actions_memory, dtype=float)
    drifted = np.asarray(environment._final_weights, dtype=float)
    steps = max(len(values) - 1, 1)
    turnover = (
        float(np.abs(chosen[1:] - drifted[:-1]).sum(axis=1).mean())
        if len(chosen) > 1
        else np.nan
    )
    reward_per_day = float(
        env_kwargs["reward_scaling"] * np.log(values[-1] / values[0]) / steps
    )
    metrics = performance_metrics_from_values(
        values,
        turnover=turnover,
        reward_per_day=reward_per_day,
    )
    return {
        "values": values,
        "dates": dates,
        "actions": chosen,
        "weights": chosen,
        "drifted_weights": drifted,
        "weight_labels": ["Cash", *list(environment._tic_list)],
        "max_weight": float(chosen[1:].max()) if len(chosen) > 1 else np.nan,
        "metrics": metrics,
    }


class PortfolioValidationCallback(BaseCallback):
    """Match the real-data control's deterministic train/validation selection."""

    def __init__(
        self,
        training_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        env_kwargs: dict[str, Any],
        eval_freq: int,
        best_model_path: Path,
        seed: int,
    ):
        super().__init__()
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.env_kwargs = env_kwargs
        self.eval_freq = int(eval_freq)
        self.best_model_path = Path(best_model_path)
        self.seed = int(seed)
        self.records: list[dict[str, float | int | bool]] = []
        self.best_score = -np.inf
        self.best_timestep = 0

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        train_result = rollout_portfolio_model(
            self.training_frame, self.model, self.env_kwargs, self.seed
        )
        validation_result = rollout_portfolio_model(
            self.validation_frame, self.model, self.env_kwargs, self.seed
        )
        train_metrics = train_result["metrics"]
        validation_metrics = validation_result["metrics"]
        score = float(validation_metrics["reward_per_day"])
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_timestep = int(self.num_timesteps)
            self.model.save(self.best_model_path)
        self.records.append(
            {
                "timesteps": int(self.num_timesteps),
                "train_reward_mean": float(train_metrics["reward_per_day"]),
                "train_reward_std": np.nan,
                "real_valid_reward": score,
                "real_valid_turnover": float(validation_metrics["turnover"]),
                "real_valid_max_weight": float(validation_result["max_weight"]),
                "is_best": bool(is_best),
                **validation_metrics,
            }
        )


class PortfolioIntrinsicMetricsCallback(BaseCallback):
    """Collect reward decomposition and PPO internals without changing training."""

    def __init__(self, log_every: int):
        super().__init__()
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self.log_every = int(log_every)
        self.records: list[dict[str, float | int | bool]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        reward_train = float(rewards.mean()) if rewards.size else np.nan
        record: dict[str, float | int | bool] = {
            "timesteps": int(self.num_timesteps),
            "reward_train": reward_train,
        }
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
            "intrinsic_surprise_center",
            "intrinsic_surprise_std",
            "intrinsic_surprise_z",
        )
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
            record[output_key] = float(logger_values.get(logger_key, np.nan))
        self.records.append(record)
        return True


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _serializable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _serializable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def portfolio_frame_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["date", "tic"], kind="mergesort").reset_index(
        drop=True
    )
    payload = {
        "columns": list(ordered.columns),
        "dtypes": ordered.dtypes.astype(str).to_dict(),
        "values": hashlib.sha256(
            pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
        ).hexdigest(),
    }
    return _content_hash(payload)


def portfolio_training_cache_key(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    variant: str,
    commission_name: str,
    commission: float,
    total_timesteps: int,
    eval_freq: int,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict[str, Any],
    ppo_kwargs: dict[str, Any],
    replay_capacity: int = MAX_INTRINSIC_REPLAY_CAPACITY,
) -> tuple[str, dict[str, Any]]:
    reward_config = resolve_variant_reward_config(train, variant, env_kwargs)
    payload = {
        "schema": TRAIN_CACHE_SCHEMA,
        "algorithm": "PPO",
        "policy": "MultiInputPolicy",
        "data_pipeline": "real_train_max_price_scale_by_ticker",
        "initial_allocation": "cash",
        "train_fingerprint": portfolio_frame_fingerprint(train),
        "validation_fingerprint": portfolio_frame_fingerprint(validation),
        "variant": variant,
        "variant_weights": VARIANT_WEIGHTS[variant],
        "commission_name": commission_name,
        "commission": float(commission),
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq),
        "seed": int(seed),
        "warmup_steps": int(warmup_steps),
        "intrinsic_reward": (
            None
            if variant == "baseline"
            else {
                "nominal_alpha": reward_config["nominal_alpha"],
                "effective_alpha": reward_config["effective_alpha"],
                "beta": reward_config["beta"],
                "surprise_mode": reward_config["surprise_mode"],
                "dimension_adjust_alpha": reward_config["dimension_adjust_alpha"],
                "observation_dim": reward_config["observation_dim"],
                "batch_size": min(128, max(2, warmup_steps)),
                "replay_capacity": min(replay_capacity, max(10_000, total_timesteps)),
                "observation": "flatten(state_then_last_action)",
                "action": "raw_symmetric_policy_action",
                "device": str(ppo_kwargs.get("device", "auto")),
            }
        ),
        "environment": {
            key: value
            for key, value in env_kwargs.items()
            if key not in {"cwd", "plot_on_terminal"}
        },
        "ppo": ppo_kwargs,
        "versions": {
            "stable_baselines3": sb3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return _content_hash(payload), _serializable(payload)


def _cache_complete(run_dir: Path, variant: str, cache_key: str) -> bool:
    required = {
        "best_model.zip",
        "last_model.zip",
        "training_log.csv",
        "validation_log.csv",
        "status.json",
    }
    if variant != "baseline":
        required.add("intrinsic_reward.pt")
    if not all((run_dir / filename).is_file() for filename in required):
        return False
    try:
        status = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(
        status.get("completed")
        and status.get("schema") == TRAIN_CACHE_SCHEMA
        and status.get("cache_key") == cache_key
    )


def plan_portfolio_training(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    variant: str,
    commission_name: str,
    commission: float,
    total_timesteps: int,
    eval_freq: int,
    seed: int,
    warmup_steps: int,
    env_kwargs: dict[str, Any],
    ppo_kwargs: dict[str, Any],
    cache_dir: Path,
    use_cache: bool = True,
    force_retrain: bool = False,
    replay_capacity: int = MAX_INTRINSIC_REPLAY_CAPACITY,
) -> PortfolioTrainingPlan:
    cache_key, cache_payload = portfolio_training_cache_key(
        train=train,
        validation=validation,
        variant=variant,
        commission_name=commission_name,
        commission=commission,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        seed=seed,
        warmup_steps=warmup_steps,
        env_kwargs=env_kwargs,
        ppo_kwargs=ppo_kwargs,
        replay_capacity=replay_capacity,
    )
    run_dir = Path(cache_dir) / cache_key
    cache_hit = bool(
        use_cache and not force_retrain and _cache_complete(run_dir, variant, cache_key)
    )
    return PortfolioTrainingPlan(
        variant=variant,
        commission_name=commission_name,
        commission=float(commission),
        seed=int(seed),
        cache_key=cache_key,
        cache_payload=cache_payload,
        run_dir=run_dir,
        cache_hit=cache_hit,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def train_portfolio_variant(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    plan: PortfolioTrainingPlan,
    total_timesteps: int,
    eval_freq: int,
    warmup_steps: int,
    env_kwargs: dict[str, Any],
    ppo_kwargs: dict[str, Any],
    show_progress: bool = True,
    replay_capacity: int = MAX_INTRINSIC_REPLAY_CAPACITY,
) -> PortfolioVariantRun:
    set_global_seed(plan.seed)
    if plan.cache_hit:
        model = PPO.load(
            plan.run_dir / "best_model",
            device=ppo_kwargs.get("device", "auto"),
        )
        status = json.loads((plan.run_dir / "status.json").read_text())
        return PortfolioVariantRun(
            variant=plan.variant,
            commission_name=plan.commission_name,
            commission=plan.commission,
            seed=plan.seed,
            model=model,
            environment=None,
            intrinsic_log=_read_csv(plan.run_dir / "training_log.csv"),
            validation_log=_read_csv(plan.run_dir / "validation_log.csv"),
            best_timestep=int(status["best_timestep"]),
            run_dir=plan.run_dir,
            cache_hit=True,
            cache_key=plan.cache_key,
        )

    environment = build_portfolio_training_environment(
        train=train,
        variant=plan.variant,
        total_timesteps=total_timesteps,
        seed=plan.seed,
        warmup_steps=warmup_steps,
        env_kwargs=env_kwargs,
        device=str(ppo_kwargs.get("device", "auto")),
        replay_capacity=replay_capacity,
    )
    monitored_environment = Monitor(environment)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    episode_logger = EpisodeLogger()
    validator = PortfolioValidationCallback(
        training_frame=train,
        validation_frame=validation,
        env_kwargs=env_kwargs,
        eval_freq=eval_freq,
        best_model_path=plan.run_dir / "best_model",
        seed=plan.seed,
    )
    metrics_logger = PortfolioIntrinsicMetricsCallback(
        log_every=min(256, max(1, total_timesteps // 200))
    )
    callbacks: list[BaseCallback] = [episode_logger, validator, metrics_logger]
    if show_progress:
        callbacks.append(
            TqdmTrainingCallback(
                total_timesteps,
                f"{plan.variant} | {plan.commission_name} | seed {plan.seed}",
            )
        )

    model = PPO(
        "MultiInputPolicy",
        monitored_environment,
        seed=plan.seed,
        **ppo_kwargs,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
    )
    model.save(plan.run_dir / "last_model")
    training_log = pd.DataFrame(metrics_logger.records)
    validation_log = pd.DataFrame(validator.records)
    training_log.to_csv(plan.run_dir / "training_log.csv", index=False)
    validation_log.to_csv(plan.run_dir / "validation_log.csv", index=False)
    pd.DataFrame(episode_logger.records).to_csv(
        plan.run_dir / "episode_log.csv", index=False
    )
    if isinstance(environment, IntrinsicRewardPortfolioOptimizationEnv):
        environment.save_intrinsic_state(plan.run_dir / "intrinsic_reward.pt")

    status = {
        "completed": True,
        "schema": TRAIN_CACHE_SCHEMA,
        "cache_key": plan.cache_key,
        "cache_payload": plan.cache_payload,
        "variant": plan.variant,
        "commission_name": plan.commission_name,
        "commission": plan.commission,
        "seed": plan.seed,
        "best_timestep": validator.best_timestep,
        "best_real_validation_reward": validator.best_score,
        "total_timesteps": total_timesteps,
    }
    (plan.run_dir / "status.json").write_text(
        json.dumps(_serializable(status), indent=2, sort_keys=True) + "\n"
    )
    best_model = PPO.load(
        plan.run_dir / "best_model",
        device=ppo_kwargs.get("device", "auto"),
    )
    return PortfolioVariantRun(
        variant=plan.variant,
        commission_name=plan.commission_name,
        commission=plan.commission,
        seed=plan.seed,
        model=best_model,
        environment=None,
        intrinsic_log=training_log,
        validation_log=validation_log,
        best_timestep=validator.best_timestep,
        run_dir=plan.run_dir,
        cache_hit=False,
        cache_key=plan.cache_key,
    )


def evaluate_portfolio_periods(
    periods: dict[str, pd.DataFrame],
    runs: Sequence[PortfolioVariantRun],
    *,
    cwd: str | Path,
    initial_amount: int = 100_000,
    time_window: int = 50,
    reward_scaling: float = 100.0,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    pd.DataFrame,
    dict[str, dict[str, dict[str, Any]]],
]:
    required = {"train", "validation", "test"}
    missing = sorted(required - set(periods))
    if missing:
        raise ValueError(f"periods is missing {missing}")
    commission_pairs = sorted({(run.commission_name, run.commission) for run in runs})
    ppo_rows = []
    ppo_results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    benchmark_rows = []
    benchmark_results: dict[str, dict[str, dict[str, Any]]] = {}
    for commission_name, commission in commission_pairs:
        kwargs = portfolio_environment_kwargs(
            commission,
            cwd=cwd,
            initial_amount=initial_amount,
            time_window=time_window,
            reward_scaling=reward_scaling,
        )
        commission_runs = [
            run for run in runs if run.commission_name == commission_name
        ]
        ppo_results[commission_name] = {}
        benchmark_results[commission_name] = {}
        for period in ("train", "validation", "test"):
            frame = periods[period]
            period_results = {}
            for run in commission_runs:
                key = f"{run.variant}__seed_{run.seed}"
                result = rollout_portfolio_model(frame, run.model, kwargs, run.seed)
                period_results[key] = result
                ppo_rows.append(
                    {
                        "period": period,
                        "commission_name": commission_name,
                        "commission": commission,
                        "variant": run.variant,
                        "seed": run.seed,
                        "best_timestep": run.best_timestep,
                        **result["metrics"],
                    }
                )
            ppo_results[commission_name][period] = period_results
            benchmark = rollout_portfolio_model(frame, None, kwargs, seed=0)
            benchmark_results[commission_name][period] = benchmark
            benchmark_rows.append(
                {
                    "period": period,
                    "commission_name": commission_name,
                    "commission": commission,
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


def finite_intrinsic_info(info: dict[str, Any]) -> bool:
    """Return whether every reward diagnostic in an info dictionary is finite."""
    return all(
        math.isfinite(float(value))
        for key, value in info.items()
        if key.startswith("reward_") or key.startswith("intrinsic_")
    )
