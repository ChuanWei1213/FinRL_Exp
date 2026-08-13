"""Reusable Synthetic-vs-Real portfolio experiment pipeline.

The module keeps long-running training independent from Jupyter while preserving
the protocol used by ``FinRL_PortfolioOptimization_PPO_Synthetic_vs_Real``:

* real-only, synthetic-only, and equally balanced real+synthetic training;
* uniform episode sampling across paths within a synthetic dataset;
* checkpoint selection on the matching real validation period; and
* final evaluation on real, out-of-sample test periods.

Synthetic files are discovered below
``data/synthetic/{ticker_group}/{dataset_id}/{model}/path*.csv``.  A trading-day
rolling schedule slices the shared long dataset into fixed train, validation, and
test windows without duplicating source files.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import glob
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm

from finrl.experiments.notebook_utils import EIIEExtractor
from finrl.experiments.notebook_utils import EpisodeLogger
from finrl.experiments.notebook_utils import PriceScaleByTicker
from finrl.experiments.notebook_utils import TqdmTrainingCallback
from finrl.experiments.notebook_utils import check_calendar_coverage
from finrl.experiments.notebook_utils import performance_metrics_from_values
from finrl.experiments.notebook_utils import read_long_market_csv
from finrl.experiments.notebook_utils import read_real_ohlcv_csvs
from finrl.experiments.notebook_utils import set_global_seed
from finrl.experiments.notebook_utils import slice_evaluation_with_lookback
from finrl.experiments.notebook_utils import slice_period
from finrl.experiments.notebook_utils import validate_long_market_frame
from finrl.experiments.notebook_utils import validate_periods
from finrl.meta.env_portfolio_optimization.env_portfolio_multipath import MultiPathEnv
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
    PortfolioOptimizationGymnasiumEnv,
)
from training_cache_fingerprint import TRAINING_DATA_FINGERPRINT_SCHEMA
from training_cache_fingerprint import fingerprint_training_sources


PRICE_COLUMNS = ("close", "high", "low")
REQUIRED_COLUMNS = ("date", "tic", *PRICE_COLUMNS)
TRAIN_CACHE_SCHEMA = "ppo_eiie_training_cache_v3"
DEFAULT_SYNTHETIC_MODEL_LABELS = {
    "armd__single_path": "ARMD (single path)",
    "cndiff__50_paths": "CN-Diff (50 paths)",
    "dva__mean_of_50_paths": "DVA (mean of 50 paths)",
    "dva__50_paths": "DVA (50 paths)",
    "nsdiff__50_paths": "NS-Diff (50 paths)",
}
DEFAULT_RUN_MODES = {
    "smoke": {
        "seeds": [0, 1],
        "total_timesteps": 4_096,
        "eval_freq": 2_048,
        "commissions": {"with_fee": 0.0025},
    },
    "full": {
        "seeds": [0, 1, 2],
        "total_timesteps": 200_000,
        "eval_freq": 20_000,
        "commissions": {"no_fee": 0.0, "with_fee": 0.0025},
    },
}


def _date_text(value: str) -> str:
    return pd.Timestamp(value).date().isoformat()


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
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _serializable(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TimeSplit:
    """One chronological train, validation, and test experiment window."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError(
                "TimeSplit.name must contain only letters, numbers, '.', '_', or '-'"
            )
        normalized = {
            field: _date_text(getattr(self, field))
            for field in (
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
            )
        }
        for field, value in normalized.items():
            object.__setattr__(self, field, value)
        validate_periods(
            (
                ("train", self.train_start, self.train_end),
                ("validation", self.val_start, self.val_end),
                ("test", self.test_start, self.test_end),
            )
        )

    @property
    def synthetic_period_id(self) -> str:
        """Directory containing synthetic paths for ``[train_start, val_end)``."""
        return f"{self.train_start}_to_{self.val_end}"

    def with_test(self, *, test_start: str, test_end: str) -> "TimeSplit":
        """Return an otherwise identical split with a different real test period."""
        return TimeSplit(
            name=self.name,
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=test_start,
            test_end=test_end,
        )


@dataclass(frozen=True)
class RollingSchedule:
    """Trading-day walk-forward schedule derived from a synthetic dataset grid."""

    name: str
    train_days: int
    validation_days: int
    test_days: int
    step_days: int
    include_partial_final_test: bool = True

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError(
                "RollingSchedule.name must contain only letters, numbers, '.', "
                "'_', or '-'"
            )
        for field in ("train_days", "validation_days", "test_days", "step_days"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")


@dataclass(frozen=True)
class RunSettings:
    mode: str
    seeds: tuple[int, ...]
    total_timesteps: int
    eval_freq: int
    commissions: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("At least one training seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Training seeds must be unique")
        if self.total_timesteps <= 0 or self.eval_freq <= 0:
            raise ValueError("total_timesteps and eval_freq must be positive")
        if not self.commissions:
            raise ValueError("At least one commission setting is required")
        if any(value < 0 for _, value in self.commissions):
            raise ValueError("Commission values must be non-negative")

    @property
    def commission_map(self) -> dict[str, float]:
        return dict(self.commissions)


@dataclass(frozen=True)
class ExperimentConfig:
    real_data_glob: str
    synthetic_root: str
    artifact_dir: str
    ticker_groups: Mapping[str, tuple[str, ...]]
    active_ticker_groups: tuple[str, ...]
    synthetic_datasets: Mapping[str, str]
    time_splits: tuple[TimeSplit, ...]
    rolling_schedule: RollingSchedule | None
    evaluation_modes: tuple[str, ...]
    synthetic_model_labels: Mapping[str, str]
    run_modes: Mapping[str, Mapping[str, Any]]
    time_window: int = 50
    initial_amount: int = 100_000
    reward_scaling: float = 100.0
    max_train_evaluation_paths: int = 3
    max_synthetic_validation_paths: int = 3
    device: str = "cpu"

    def __post_init__(self) -> None:
        normalized_groups: dict[str, tuple[str, ...]] = {}
        for raw_name, raw_tickers in self.ticker_groups.items():
            name = str(raw_name)
            if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError(
                    "Ticker-group names must contain only letters, numbers, "
                    "'.', '_', or '-'"
                )
            tickers = tuple(sorted(str(ticker).upper() for ticker in raw_tickers))
            if not tickers:
                raise ValueError(f"Ticker group {name!r} must not be empty")
            if len(set(tickers)) != len(tickers):
                raise ValueError(f"Ticker group {name!r} contains duplicates")
            normalized_groups[name] = tickers
        if not normalized_groups:
            raise ValueError("At least one ticker group is required")
        object.__setattr__(self, "ticker_groups", normalized_groups)
        active = tuple(str(name) for name in self.active_ticker_groups)
        if not active:
            raise ValueError("At least one active ticker group is required")
        if len(set(active)) != len(active):
            raise ValueError("Active ticker groups must be unique")
        missing_groups = sorted(set(active) - set(normalized_groups))
        if missing_groups:
            raise ValueError(f"Unknown active ticker group(s): {missing_groups}")
        object.__setattr__(self, "active_ticker_groups", active)
        normalized_datasets = {
            str(group): str(dataset_id)
            for group, dataset_id in self.synthetic_datasets.items()
        }
        missing_datasets = sorted(set(normalized_groups) - set(normalized_datasets))
        if missing_datasets:
            raise ValueError(
                f"Missing synthetic dataset IDs for ticker group(s): {missing_datasets}"
            )
        for group, dataset_id in normalized_datasets.items():
            if group not in normalized_groups:
                raise ValueError(
                    f"Synthetic dataset references unknown group {group!r}"
                )
            if not dataset_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_id):
                raise ValueError(f"Invalid synthetic dataset ID {dataset_id!r}")
        object.__setattr__(self, "synthetic_datasets", normalized_datasets)
        if not self.time_splits and self.rolling_schedule is None:
            raise ValueError("Configure either time_splits or rolling_schedule")
        names = [split.name for split in self.time_splits]
        if len(set(names)) != len(names):
            raise ValueError(f"Time split names must be unique: {names}")
        modes = tuple(str(mode).lower() for mode in self.evaluation_modes)
        allowed_modes = {"independent", "continuous"}
        unknown_modes = sorted(set(modes) - allowed_modes)
        if unknown_modes or not modes:
            raise ValueError(
                f"evaluation_modes must use {sorted(allowed_modes)}, got {modes}"
            )
        if len(set(modes)) != len(modes):
            raise ValueError("evaluation_modes must be unique")
        if (
            "continuous" in modes
            and self.rolling_schedule is not None
            and self.rolling_schedule.step_days != self.rolling_schedule.test_days
        ):
            raise ValueError("Continuous evaluation requires step_days == test_days")
        object.__setattr__(self, "evaluation_modes", modes)
        if self.time_window <= 1:
            raise ValueError("time_window must be greater than one")
        if self.initial_amount <= 0 or self.reward_scaling <= 0:
            raise ValueError("initial_amount and reward_scaling must be positive")
        if self.max_train_evaluation_paths <= 0:
            raise ValueError("max_train_evaluation_paths must be positive")
        if self.max_synthetic_validation_paths <= 0:
            raise ValueError("max_synthetic_validation_paths must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        split_rows = raw.get("time_splits", ())
        time_splits = tuple(TimeSplit(**row) for row in split_rows)
        rolling_raw = raw.get("rolling_schedule")
        rolling_schedule = (
            RollingSchedule(**rolling_raw) if rolling_raw is not None else None
        )
        labels = dict(DEFAULT_SYNTHETIC_MODEL_LABELS)
        labels.update(raw.get("synthetic_model_labels", {}))
        run_modes = dict(DEFAULT_RUN_MODES)
        run_modes.update(raw.get("run_modes", {}))
        raw_ticker_groups = raw.get("ticker_groups")
        if raw_ticker_groups is None:
            ticker_groups = {"default": tuple(raw.get("tickers", ()))}
            active_ticker_groups = ("default",)
        else:
            if not isinstance(raw_ticker_groups, Mapping):
                raise ValueError("ticker_groups must contain a JSON object")
            ticker_groups = {
                str(name): tuple(tickers) for name, tickers in raw_ticker_groups.items()
            }
            active_ticker_groups = tuple(
                raw.get("active_ticker_groups", tuple(ticker_groups))
            )
        return cls(
            real_data_glob=str(raw.get("real_data_glob", "data/real/*.csv")),
            synthetic_root=str(raw.get("synthetic_root", "data/synthetic")),
            artifact_dir=str(raw.get("artifact_dir", "results/ppo_synthetic_vs_real")),
            ticker_groups=ticker_groups,
            active_ticker_groups=active_ticker_groups,
            synthetic_datasets=dict(raw.get("synthetic_datasets", {})),
            time_splits=time_splits,
            rolling_schedule=rolling_schedule,
            evaluation_modes=tuple(
                raw.get("evaluation_modes", ("independent", "continuous"))
            ),
            synthetic_model_labels=labels,
            run_modes=run_modes,
            time_window=int(raw.get("time_window", 50)),
            initial_amount=int(raw.get("initial_amount", 100_000)),
            reward_scaling=raw.get("reward_scaling", 100),
            max_train_evaluation_paths=int(raw.get("max_train_evaluation_paths", 3)),
            max_synthetic_validation_paths=int(
                raw.get("max_synthetic_validation_paths", 3)
            ),
            device=str(raw.get("device", "cpu")),
        )

    def resolve_run_settings(self, mode: str) -> RunSettings:
        normalized = mode.lower()
        if normalized not in self.run_modes:
            raise ValueError(
                f"Unknown run mode {mode!r}; expected one of {sorted(self.run_modes)}"
            )
        raw = self.run_modes[normalized]
        commissions = raw.get("commissions", {})
        return RunSettings(
            mode=normalized,
            seeds=tuple(int(seed) for seed in raw["seeds"]),
            total_timesteps=int(raw["total_timesteps"]),
            eval_freq=int(raw["eval_freq"]),
            commissions=tuple(
                (str(name), float(value)) for name, value in commissions.items()
            ),
        )

    def select_splits(self, names: Sequence[str] | None) -> tuple[TimeSplit, ...]:
        if not names:
            return self.time_splits
        requested = list(dict.fromkeys(names))
        by_name = {split.name: split for split in self.time_splits}
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ValueError(f"Unknown time split(s): {missing}")
        return tuple(by_name[name] for name in requested)

    def select_ticker_groups(
        self, names: Sequence[str] | None
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        requested = (
            list(dict.fromkeys(names)) if names else list(self.active_ticker_groups)
        )
        missing = sorted(set(requested) - set(self.ticker_groups))
        if missing:
            raise ValueError(f"Unknown ticker group(s): {missing}")
        return tuple((name, self.ticker_groups[name]) for name in requested)

    def tickers_for(self, ticker_group: str) -> tuple[str, ...]:
        try:
            return self.ticker_groups[ticker_group]
        except KeyError as error:
            raise ValueError(f"Unknown ticker group {ticker_group!r}") from error

    def dataset_id_for(self, ticker_group: str) -> str:
        try:
            return self.synthetic_datasets[ticker_group]
        except KeyError as error:
            raise ValueError(
                f"No synthetic dataset configured for ticker group {ticker_group!r}"
            ) from error

    @property
    def tickers(self) -> tuple[str, ...]:
        """Legacy accessor for configs with exactly one active ticker group."""
        if len(self.active_ticker_groups) != 1:
            raise ValueError(
                "config.tickers is ambiguous with multiple active ticker groups; "
                "use config.tickers_for(name)"
            )
        return self.tickers_for(self.active_ticker_groups[0])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "real_data_glob": self.real_data_glob,
            "synthetic_root": self.synthetic_root,
            "artifact_dir": self.artifact_dir,
            "ticker_groups": {
                name: list(tickers) for name, tickers in self.ticker_groups.items()
            },
            "active_ticker_groups": list(self.active_ticker_groups),
            "synthetic_datasets": dict(self.synthetic_datasets),
            "time_splits": [asdict(split) for split in self.time_splits],
            "rolling_schedule": (
                asdict(self.rolling_schedule) if self.rolling_schedule else None
            ),
            "evaluation_modes": list(self.evaluation_modes),
            "synthetic_model_labels": dict(self.synthetic_model_labels),
            "run_modes": _serializable(dict(self.run_modes)),
            "time_window": self.time_window,
            "initial_amount": self.initial_amount,
            "reward_scaling": self.reward_scaling,
            "max_train_evaluation_paths": self.max_train_evaluation_paths,
            "max_synthetic_validation_paths": self.max_synthetic_validation_paths,
            "device": self.device,
        }


@dataclass
class PreparedWindow:
    split: TimeSplit
    ticker_group: str
    dataset_id: str
    tickers: tuple[str, ...]
    real_files: list[Path]
    synthetic_files_by_model: dict[str, list[Path]]
    real_pipeline: dict[str, pd.DataFrame]
    synthetic_pipelines: dict[str, dict[str, Any]]
    training_data_fingerprints: dict[str, str]
    data_fingerprint: str
    data_summary: pd.DataFrame
    scale_summary: pd.DataFrame

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.synthetic_files_by_model)


@dataclass(frozen=True)
class ExperimentPaths:
    artifact_root: Path
    output_root: Path
    experiment_root: Path
    train_cache_root: Path
    experiment_id: str


@dataclass(frozen=True)
class TrainingPlan:
    group: str
    commission_name: str
    commission: float
    seed: int
    identifier: str
    cache_key: str
    cache_config: dict[str, Any]
    planned_run_dir: Path
    cache_match: tuple[Path, dict[str, Any]] | None


@dataclass
class WindowResult:
    split: TimeSplit
    ticker_group: str
    dataset_id: str
    tickers: tuple[str, ...]
    experiment_root: Path
    experiment_id: str
    run_table: pd.DataFrame
    test_metrics: pd.DataFrame
    aggregate_summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    representative_runs: pd.DataFrame
    prepared: PreparedWindow
    run_records: list[dict[str, Any]]


@dataclass
class ExperimentSuiteResult:
    manifest_path: Path
    suite_root: Path
    window_results: list[WindowResult]
    test_metrics: pd.DataFrame
    aggregate_summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    continuous_metrics: pd.DataFrame


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must contain a JSON object")
    return ExperimentConfig.from_mapping(raw)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def synthetic_group(model_name: str) -> str:
    return f"synthetic::{model_name}"


def real_synthetic_group(model_name: str) -> str:
    return f"real_synthetic::{model_name}"


def is_real_synthetic_group(group: str) -> bool:
    return group.startswith("real_synthetic::")


def model_from_group(group: str) -> str | None:
    for prefix in ("synthetic::", "real_synthetic::"):
        if group.startswith(prefix):
            return group[len(prefix) :]
    return None


def training_groups(model_names: Sequence[str]) -> list[str]:
    return [
        "real_trained",
        *[
            group
            for model_name in model_names
            for group in (
                synthetic_group(model_name),
                real_synthetic_group(model_name),
            )
        ],
    ]


def synthetic_model_label(config: ExperimentConfig, model_name: str) -> str:
    return config.synthetic_model_labels.get(model_name, model_name)


def training_group_label(
    config: ExperimentConfig, group: str, split: TimeSplit | None = None
) -> str:
    if group == "real_trained":
        return "Real data" if split is None else f"Real data ({split.name})"
    model_name = model_from_group(group)
    label = synthetic_model_label(config, model_name or group)
    return f"Real + {label}" if is_real_synthetic_group(group) else label


def synthetic_dataset_directory(
    project_root: Path,
    config: ExperimentConfig,
    ticker_group: str,
) -> Path:
    return (
        resolve_project_path(project_root, config.synthetic_root)
        / ticker_group
        / config.dataset_id_for(ticker_group)
    )


def discover_synthetic_datasets(
    project_root: Path,
    config: ExperimentConfig,
    ticker_group: str,
) -> dict[str, list[Path]]:
    dataset_root = synthetic_dataset_directory(project_root, config, ticker_group)
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Synthetic dataset directory not found: "
            f"{dataset_root}. Expected data/synthetic/<ticker_group>/"
            "<dataset_id>/<model>/path*.csv"
        )
    discovered: dict[str, list[Path]] = {}
    for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        paths = sorted(model_dir.glob("path*.csv"))
        if paths:
            discovered[model_dir.name] = paths
    if not discovered:
        raise FileNotFoundError(
            f"No synthetic paths found below {dataset_root}/<dataset>/path*.csv"
        )
    return {
        model_name: discovered[model_name]
        for model_name in config.synthetic_model_labels
        if model_name in discovered
    }


def _synthetic_dataset_dates(
    *,
    project_root: Path,
    config: ExperimentConfig,
    ticker_group: str,
    tickers: Sequence[str],
) -> pd.DatetimeIndex:
    discovered = discover_synthetic_datasets(project_root, config, ticker_group)
    first_model = next(iter(discovered))
    first_path = discovered[first_model][0]
    grid = pd.read_csv(first_path, usecols=["date", "tic"])
    grid["date"] = pd.to_datetime(grid["date"], errors="raise").dt.normalize()
    grid["tic"] = grid["tic"].astype(str).str.upper()
    if grid.duplicated(["date", "tic"]).any():
        raise ValueError(f"{first_path}: duplicate (date, tic) rows")
    actual_tickers = tuple(sorted(grid["tic"].unique()))
    expected_tickers = tuple(sorted(tickers))
    if actual_tickers != expected_tickers:
        raise ValueError(
            f"{first_path}: expected tickers {expected_tickers}, got {actual_tickers}"
        )
    counts = grid.groupby("date")["tic"].nunique()
    if not (counts == len(expected_tickers)).all():
        raise ValueError(f"{first_path}: incomplete ticker grid")
    return pd.DatetimeIndex(sorted(grid["date"].unique()))


def rolling_time_splits(
    *,
    project_root: Path,
    config: ExperimentConfig,
    ticker_group: str,
    tickers: Sequence[str],
    names: Sequence[str] | None = None,
) -> tuple[TimeSplit, ...]:
    """Derive fixed-length trading-day windows from the selected dataset."""
    if config.rolling_schedule is None:
        return config.select_splits(names)
    schedule = config.rolling_schedule
    dates = _synthetic_dataset_dates(
        project_root=project_root,
        config=config,
        ticker_group=ticker_group,
        tickers=tickers,
    )
    history_days = schedule.train_days + schedule.validation_days
    if len(dates) <= history_days:
        raise ValueError(
            f"Synthetic dataset {config.dataset_id_for(ticker_group)!r} has "
            f"{len(dates)} dates; rolling schedule needs more than {history_days}"
        )
    splits: list[TimeSplit] = []
    test_start_index = history_days
    window_index = 0
    while test_start_index < len(dates):
        test_stop_index = min(test_start_index + schedule.test_days, len(dates))
        test_count = test_stop_index - test_start_index
        if test_count < schedule.test_days and not schedule.include_partial_final_test:
            break
        train_start_index = test_start_index - history_days
        train_end_index = train_start_index + schedule.train_days
        test_end = (
            dates[test_stop_index]
            if test_stop_index < len(dates)
            else dates[-1] + pd.Timedelta(days=1)
        )
        splits.append(
            TimeSplit(
                name=(
                    f"{schedule.name}_{window_index:03d}__test_"
                    f"{dates[test_start_index]:%Y%m%d}_"
                    f"{dates[test_stop_index - 1]:%Y%m%d}"
                ),
                train_start=dates[train_start_index].date().isoformat(),
                train_end=dates[train_end_index].date().isoformat(),
                val_start=dates[train_end_index].date().isoformat(),
                val_end=dates[test_start_index].date().isoformat(),
                test_start=dates[test_start_index].date().isoformat(),
                test_end=test_end.date().isoformat(),
            )
        )
        window_index += 1
        test_start_index += schedule.step_days
    if not splits:
        raise ValueError("Rolling schedule produced no test windows")
    if not names:
        return tuple(splits)
    requested = list(dict.fromkeys(names))
    by_name = {split.name: split for split in splits}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"Unknown rolling window(s): {missing}")
    return tuple(by_name[name] for name in requested)


def load_real_market_data(
    project_root: Path,
    config: ExperimentConfig,
    tickers: Sequence[str] | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    selected_tickers = tuple(tickers or config.tickers)
    real_pattern = str(resolve_project_path(project_root, config.real_data_glob))
    ticker_set = set(selected_tickers)
    real_files = [
        Path(path)
        for path in sorted(glob.glob(real_pattern))
        if Path(path).stem.upper() in ticker_set
    ]
    real_ohlcv = read_real_ohlcv_csvs(real_files, selected_tickers)
    frame = validate_long_market_frame(
        real_ohlcv[list(REQUIRED_COLUMNS)],
        "real data",
        selected_tickers,
        REQUIRED_COLUMNS,
        PRICE_COLUMNS,
    )
    return real_files, frame


def _read_synthetic_path(
    path: Path,
    tickers: Sequence[str],
    label: str,
) -> pd.DataFrame:
    return read_long_market_csv(
        path,
        label,
        tickers,
        REQUIRED_COLUMNS,
        PRICE_COLUMNS,
    )


def _fingerprint_files(project_root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            identity = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            identity = path.name
        digest.update(identity.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _scaled_range(
    frame: pd.DataFrame, price_columns: Sequence[str]
) -> tuple[float, float]:
    values = frame[list(price_columns)].to_numpy(dtype=float)
    return float(values.min()), float(values.max())


def _execution_dates(frame: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.to_datetime(frame["date"])
    selected = dates[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))]
    return pd.DatetimeIndex(sorted(selected.unique()))


def _require_same_dates(
    expected: pd.DatetimeIndex,
    actual: pd.DatetimeIndex,
    *,
    label: str,
) -> None:
    if expected.equals(actual):
        return
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    raise ValueError(
        f"{label}: trading-day grid differs; "
        f"missing={[date.date().isoformat() for date in missing[:5]]}, "
        f"extra={[date.date().isoformat() for date in extra[:5]]}"
    )


def prepare_window_data(
    *,
    project_root: Path,
    config: ExperimentConfig,
    split: TimeSplit,
    real_files: Sequence[Path],
    real_raw: pd.DataFrame,
    ticker_group: str | None = None,
    tickers: Sequence[str] | None = None,
) -> PreparedWindow:
    """Load, validate, slice, and scale one multi-source experiment window."""
    selected_group = ticker_group or config.active_ticker_groups[0]
    selected_tickers = tuple(tickers or config.tickers_for(selected_group))
    dataset_id = config.dataset_id_for(selected_group)
    if selected_tickers != config.tickers_for(selected_group):
        raise ValueError(
            f"Ticker list for {selected_group!r} does not match the experiment config"
        )
    check_calendar_coverage(
        real_raw,
        split.train_start,
        split.val_end,
        f"{split.name} real train+validation",
    )
    check_calendar_coverage(
        real_raw,
        split.test_start,
        split.test_end,
        f"{split.name} real test",
    )
    synthetic_files = discover_synthetic_datasets(project_root, config, selected_group)
    unknown_labels = sorted(set(synthetic_files) - set(config.synthetic_model_labels))
    if unknown_labels:
        raise ValueError(
            "Add reader-facing labels to synthetic_model_labels for: "
            f"{unknown_labels}"
        )

    synthetic_raw: dict[str, list[pd.DataFrame]] = {}
    for model_name, paths in synthetic_files.items():
        frames = []
        for path in paths:
            label = (
                f"synthetic {split.name}/{selected_group}/{dataset_id}/"
                f"{model_name}/{path.name}"
            )
            frame = _read_synthetic_path(path, selected_tickers, label)
            check_calendar_coverage(
                frame,
                split.train_start,
                split.val_end,
                label,
            )
            frames.append(frame)
        synthetic_raw[model_name] = frames

    real_train_raw = slice_period(
        real_raw,
        split.train_start,
        split.train_end,
        f"{split.name} real train",
        minimum_dates=config.time_window + 1,
    )
    real_valid_raw = slice_evaluation_with_lookback(
        real_raw,
        split.val_start,
        split.val_end,
        f"{split.name} real validation",
        time_window=config.time_window,
    )
    real_test_raw = slice_evaluation_with_lookback(
        real_raw,
        split.test_start,
        split.test_end,
        f"{split.name} real test",
        time_window=config.time_window,
    )

    synthetic_train_raw: dict[str, list[pd.DataFrame]] = {}
    synthetic_valid_raw: dict[str, list[pd.DataFrame]] = {}
    for model_name, frames in synthetic_raw.items():
        synthetic_train_raw[model_name] = [
            slice_period(
                frame,
                split.train_start,
                split.train_end,
                f"{split.name} synthetic {model_name} train path {index}",
                minimum_dates=config.time_window + 1,
            )
            for index, frame in enumerate(frames)
        ]
        synthetic_valid_raw[model_name] = [
            slice_evaluation_with_lookback(
                frame,
                split.val_start,
                split.val_end,
                f"{split.name} synthetic {model_name} validation path {index}",
                time_window=config.time_window,
            )
            for index, frame in enumerate(frames)
        ]

    expected_train_dates = _execution_dates(
        real_train_raw, split.train_start, split.train_end
    )
    expected_validation_dates = _execution_dates(
        real_valid_raw, split.val_start, split.val_end
    )
    expected_test_dates = _execution_dates(
        next(iter(synthetic_raw.values()))[0], split.test_start, split.test_end
    )
    _require_same_dates(
        expected_test_dates,
        _execution_dates(real_test_raw, split.test_start, split.test_end),
        label=f"{split.name} real test",
    )
    for model_name in synthetic_files:
        for path_index, frame in enumerate(synthetic_train_raw[model_name]):
            _require_same_dates(
                expected_train_dates,
                _execution_dates(frame, split.train_start, split.train_end),
                label=f"{split.name} synthetic {model_name} train path {path_index}",
            )
        for path_index, frame in enumerate(synthetic_valid_raw[model_name]):
            _require_same_dates(
                expected_validation_dates,
                _execution_dates(frame, split.val_start, split.val_end),
                label=(
                    f"{split.name} synthetic {model_name} validation path "
                    f"{path_index}"
                ),
            )

    real_scaler = PriceScaleByTicker(selected_tickers, PRICE_COLUMNS).fit(
        [real_train_raw]
    )
    real_pipeline = {
        "train": real_scaler.transform(real_train_raw),
        "real_valid": real_scaler.transform(real_valid_raw),
        "real_test": real_scaler.transform(real_test_raw),
    }
    synthetic_pipelines: dict[str, dict[str, Any]] = {}
    for model_name in synthetic_files:
        scaler = PriceScaleByTicker(selected_tickers, PRICE_COLUMNS).fit(
            synthetic_train_raw[model_name]
        )
        synthetic_pipelines[model_name] = {
            "scaler": scaler,
            "train_paths": [
                scaler.transform(frame) for frame in synthetic_train_raw[model_name]
            ],
            "synthetic_valid_paths": [
                scaler.transform(frame) for frame in synthetic_valid_raw[model_name]
            ],
            "real_train": scaler.transform(real_train_raw),
            "real_valid": scaler.transform(real_valid_raw),
            "real_test": scaler.transform(real_test_raw),
        }

    training_source_files: dict[str, list[tuple[str, Path]]] = {
        "real_trained": [("real", path) for path in real_files]
    }
    for model_name in synthetic_files:
        sources = [
            *[("real", path) for path in real_files],
            *[("synthetic", path) for path in synthetic_files[model_name]],
        ]
        training_source_files[synthetic_group(model_name)] = sources
        training_source_files[real_synthetic_group(model_name)] = sources
    fingerprints = {
        group: fingerprint_training_sources(
            project_root=project_root,
            sources=sources,
            start=split.train_start,
            end=split.val_end,
        )
        for group, sources in training_source_files.items()
    }

    data_summary_rows: list[dict[str, Any]] = [
        {
            "window_id": split.name,
            "source": "Real data",
            "split": "train",
            "paths": 1,
            "first_date": real_train_raw["date"].min().date(),
            "last_date": real_train_raw["date"].max().date(),
            "dates_per_path": real_train_raw["date"].nunique(),
        },
        {
            "window_id": split.name,
            "source": "Real data",
            "split": "validation",
            "paths": 1,
            "first_date": pd.Timestamp(split.val_start).date(),
            "last_date": real_valid_raw["date"].max().date(),
            "dates_per_path": int(
                (real_valid_raw["date"] >= pd.Timestamp(split.val_start)).sum()
                // len(selected_tickers)
            ),
        },
        {
            "window_id": split.name,
            "source": "Real data",
            "split": "test",
            "paths": 1,
            "first_date": pd.Timestamp(split.test_start).date(),
            "last_date": real_test_raw["date"].max().date(),
            "dates_per_path": int(
                (real_test_raw["date"] >= pd.Timestamp(split.test_start)).sum()
                // len(selected_tickers)
            ),
        },
    ]
    scale_summary_rows = [
        {
            "window_id": split.name,
            "source": "Real data",
            "split": period,
            "range": _scaled_range(frame, PRICE_COLUMNS),
        }
        for period, frame in real_pipeline.items()
    ]
    for model_name, pipeline in synthetic_pipelines.items():
        label = synthetic_model_label(config, model_name)
        train_paths = synthetic_train_raw[model_name]
        valid_paths = synthetic_valid_raw[model_name]
        data_summary_rows.extend(
            (
                {
                    "window_id": split.name,
                    "source": label,
                    "split": "synthetic train",
                    "paths": len(train_paths),
                    "first_date": min(x["date"].min() for x in train_paths).date(),
                    "last_date": max(x["date"].max() for x in train_paths).date(),
                    "dates_per_path": (
                        f"{min(x['date'].nunique() for x in train_paths)}–"
                        f"{max(x['date'].nunique() for x in train_paths)}"
                    ),
                },
                {
                    "window_id": split.name,
                    "source": label,
                    "split": "synthetic validation",
                    "paths": len(valid_paths),
                    "first_date": pd.Timestamp(split.val_start).date(),
                    "last_date": max(x["date"].max() for x in valid_paths).date(),
                    "dates_per_path": (
                        f"{min((x['date'] >= pd.Timestamp(split.val_start)).sum() // len(selected_tickers) for x in valid_paths)}–"
                        f"{max((x['date'] >= pd.Timestamp(split.val_start)).sum() // len(selected_tickers) for x in valid_paths)}"
                    ),
                },
            )
        )
        scale_summary_rows.extend(
            {
                "window_id": split.name,
                "source": label,
                "split": period,
                "range": _scaled_range(frame, PRICE_COLUMNS),
            }
            for period, frame in (
                ("real train", pipeline["real_train"]),
                ("real validation", pipeline["real_valid"]),
                ("real test", pipeline["real_test"]),
            )
        )
        scale_summary_rows.append(
            {
                "window_id": split.name,
                "source": label,
                "split": "synthetic train",
                "range": (
                    min(
                        _scaled_range(frame, PRICE_COLUMNS)[0]
                        for frame in pipeline["train_paths"]
                    ),
                    max(
                        _scaled_range(frame, PRICE_COLUMNS)[1]
                        for frame in pipeline["train_paths"]
                    ),
                ),
            }
        )

    all_synthetic_files = [path for paths in synthetic_files.values() for path in paths]
    data_summary = pd.DataFrame(data_summary_rows)
    data_summary.insert(1, "ticker_group", selected_group)
    scale_summary = pd.DataFrame(scale_summary_rows)
    scale_summary.insert(1, "ticker_group", selected_group)
    return PreparedWindow(
        split=split,
        ticker_group=selected_group,
        dataset_id=dataset_id,
        tickers=selected_tickers,
        real_files=list(real_files),
        synthetic_files_by_model=synthetic_files,
        real_pipeline=real_pipeline,
        synthetic_pipelines=synthetic_pipelines,
        training_data_fingerprints=fingerprints,
        data_fingerprint=_fingerprint_files(
            project_root, [*real_files, *all_synthetic_files]
        ),
        data_summary=data_summary,
        scale_summary=scale_summary,
    )


def environment_kwargs(
    config: ExperimentConfig,
    commission: float,
    *,
    cwd: str | Path,
) -> dict[str, Any]:
    if commission < 0:
        raise ValueError("commission must be non-negative")
    return {
        "initial_amount": config.initial_amount,
        "time_window": config.time_window,
        "features": list(PRICE_COLUMNS),
        "normalize_df": None,
        "reward_scaling": config.reward_scaling,
        "action_space_mode": "symmetric",
        "action_scale": 5.0,
        "return_last_action": True,
        "plot_on_terminal": False,
        "cwd": str(cwd),
        "comission_fee_pct": float(commission),
    }


def ppo_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "n_steps": 2_048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "device": config.device,
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


def _semantic_ppo_config(config: Mapping[str, Any]) -> dict[str, Any]:
    semantic = _serializable(dict(config))
    semantic.pop("device", None)
    semantic.pop("verbose", None)
    policy = semantic.get("policy_kwargs", {})
    extractor = policy.get("features_extractor_class")
    if isinstance(extractor, str):
        policy["features_extractor_class"] = extractor.rsplit(".", 1)[-1]
    return semantic


def _runtime_provenance(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "device": config.device,
        "python": sys.version.split()[0],
        "library_versions": {
            "stable_baselines3": sb3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def resolve_experiment_paths(
    *,
    project_root: Path,
    config: ExperimentConfig,
    settings: RunSettings,
    prepared: PreparedWindow,
) -> ExperimentPaths:
    artifact_root = resolve_project_path(project_root, config.artifact_dir)
    output_root = (
        artifact_root / settings.mode / prepared.split.name / prepared.ticker_group
    )
    payload = {
        "split": asdict(prepared.split),
        "ticker_group": prepared.ticker_group,
        "tickers": list(prepared.tickers),
        "synthetic_dataset_id": prepared.dataset_id,
        "synthetic_models": list(prepared.model_names),
        "data_fingerprint": prepared.data_fingerprint,
        "mode": settings.mode,
        "seeds": list(settings.seeds),
        "total_timesteps": settings.total_timesteps,
        "eval_freq": settings.eval_freq,
        "commissions": settings.commission_map,
        "time_window": config.time_window,
    }
    experiment_id = _content_hash(payload)[:12]
    return ExperimentPaths(
        artifact_root=artifact_root,
        output_root=output_root,
        experiment_root=output_root / experiment_id,
        train_cache_root=artifact_root / "train_cache",
        experiment_id=experiment_id,
    )


def training_cache_key(
    *,
    config: ExperimentConfig,
    settings: RunSettings,
    prepared: PreparedWindow,
    paths: ExperimentPaths,
    group: str,
    commission: float,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    if group not in prepared.training_data_fingerprints:
        raise KeyError(f"No training-data fingerprint for {group}")
    base_env = environment_kwargs(config, 0.0, cwd=paths.experiment_root)
    base_env.pop("comission_fee_pct")
    payload: dict[str, Any] = {
        "algorithm": "PPO",
        "policy": "MultiInputPolicy",
        "train": [prepared.split.train_start, prepared.split.train_end],
        "validation": [prepared.split.val_start, prepared.split.val_end],
        "tickers": list(prepared.tickers),
        "price_columns": list(PRICE_COLUMNS),
        "time_window": config.time_window,
        "total_timesteps": settings.total_timesteps,
        "eval_freq": settings.eval_freq,
        "max_train_evaluation_paths": config.max_train_evaluation_paths,
        "max_synthetic_validation_paths": config.max_synthetic_validation_paths,
        "environment": {
            key: value
            for key, value in base_env.items()
            if key not in {"cwd", "plot_on_terminal"}
        },
        "ppo": _semantic_ppo_config(ppo_kwargs(config)),
        "training_data_fingerprint": prepared.training_data_fingerprints[group],
        "group": group,
        "commission": float(commission),
        "seed": int(seed),
    }
    if is_real_synthetic_group(group):
        payload["episode_source_sampling"] = {
            "real": 0.5,
            "synthetic": 0.5,
            "within_synthetic": "uniform_by_path",
        }
    return _content_hash(payload), payload


def run_id(group: str, commission_name: str, seed: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", group).strip("_")
    group_hash = hashlib.sha256(group.encode()).hexdigest()[:8]
    return f"{slug}_{group_hash}__{commission_name}__seed_{seed}"


def _cache_artifact_paths(run_dir: Path) -> list[Path]:
    return [
        Path(str(run_dir / "best_model") + ".zip"),
        Path(str(run_dir / "last_model") + ".zip"),
        run_dir / "training_log.csv",
        run_dir / "validation_log.csv",
    ]


def _legacy_semantic_cache_config(status: Mapping[str, Any]) -> dict[str, Any] | None:
    legacy = status.get("training_cache_config")
    if not isinstance(legacy, dict):
        return None
    semantic = _serializable(legacy)
    semantic.pop("schema", None)
    semantic.pop("library_versions", None)
    semantic.pop("commission_name", None)
    environment = semantic.get("environment", {})
    environment.pop("cwd", None)
    environment.pop("plot_on_terminal", None)
    semantic["ppo"] = _semantic_ppo_config(semantic.get("ppo", {}))
    semantic["algorithm"] = "PPO"
    semantic["policy"] = "MultiInputPolicy"
    return semantic


def _status_semantic_cache_config(
    status: Mapping[str, Any],
) -> dict[str, Any] | None:
    semantic = status.get("semantic_cache_config")
    return (
        semantic
        if isinstance(semantic, dict)
        else _legacy_semantic_cache_config(status)
    )


def find_compatible_training_cache(
    *,
    cache_root: Path,
    cache_key: str,
    preferred_run_dir: Path,
) -> tuple[Path, dict[str, Any]] | None:
    preferred_status = preferred_run_dir / "status.json"
    candidates = [preferred_status]
    candidates.extend(
        path
        for path in sorted(cache_root.glob("*/status.json"))
        if path != preferred_status
    )
    for status_path in candidates:
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        semantic = _status_semantic_cache_config(status)
        if semantic is None or _content_hash(semantic) != cache_key:
            continue
        run_dir = status_path.parent
        if status.get("completed") is True and all(
            path.is_file() for path in _cache_artifact_paths(run_dir)
        ):
            return status_path, status
    return None


def plan_training_run(
    *,
    config: ExperimentConfig,
    settings: RunSettings,
    prepared: PreparedWindow,
    paths: ExperimentPaths,
    group: str,
    commission_name: str,
    commission: float,
    seed: int,
    use_cache: bool,
    force_retrain: bool,
) -> TrainingPlan:
    identifier = run_id(group, commission_name, seed)
    cache_key, cache_config = training_cache_key(
        config=config,
        settings=settings,
        prepared=prepared,
        paths=paths,
        group=group,
        commission=commission,
        seed=seed,
    )
    planned_run_dir = (
        paths.train_cache_root / f"{identifier}__{cache_key[:12]}"
        if use_cache
        else paths.experiment_root / "uncached_runs" / identifier
    )
    cache_match = None
    if use_cache and not force_retrain:
        cache_match = find_compatible_training_cache(
            cache_root=paths.train_cache_root,
            cache_key=cache_key,
            preferred_run_dir=planned_run_dir,
        )
    return TrainingPlan(
        group=group,
        commission_name=commission_name,
        commission=float(commission),
        seed=int(seed),
        identifier=identifier,
        cache_key=cache_key,
        cache_config=cache_config,
        planned_run_dir=planned_run_dir,
        cache_match=cache_match,
    )


def rollout(
    frame: pd.DataFrame,
    model: PPO | None,
    kwargs: Mapping[str, Any],
    seed: int,
    *,
    initial_portfolio_value: float | None = None,
    initial_actual_weights: Sequence[float] | None = None,
    initial_last_action: Sequence[float] | None = None,
) -> dict[str, Any]:
    environment = PortfolioOptimizationGymnasiumEnv(frame, **dict(kwargs))
    observation, _ = environment.reset(seed=seed)
    n_actions = environment.action_space.shape[0]
    if initial_actual_weights is not None:
        actual_weights = np.asarray(initial_actual_weights, dtype=np.float32)
        if actual_weights.shape != (n_actions,):
            raise ValueError(
                f"Initial weights shape {actual_weights.shape} != {(n_actions,)}"
            )
        if not np.isfinite(actual_weights).all() or (actual_weights < 0).any():
            raise ValueError("Initial weights must be finite and non-negative")
        if not np.isclose(actual_weights.sum(), 1.0, atol=1e-6):
            raise ValueError("Initial weights must sum to one")
        portfolio_value = float(
            kwargs["initial_amount"]
            if initial_portfolio_value is None
            else initial_portfolio_value
        )
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError("Initial portfolio value must be finite and positive")
        last_action = np.asarray(
            actual_weights if initial_last_action is None else initial_last_action,
            dtype=np.float32,
        )
        if last_action.shape != (n_actions,):
            raise ValueError(f"Initial last action shape must be {(n_actions,)}")
        if not np.isfinite(last_action).all() or (last_action < 0).any():
            raise ValueError("Initial last action must be finite and non-negative")
        if not np.isclose(last_action.sum(), 1.0, atol=1e-6):
            raise ValueError("Initial last action must sum to one")
        environment._portfolio_value = portfolio_value
        environment._asset_memory["initial"][0] = portfolio_value
        environment._asset_memory["final"][0] = portfolio_value
        environment._final_weights[0] = actual_weights.copy()
        environment._actions_memory[0] = last_action.copy()
        if isinstance(observation, dict):
            observation = dict(observation)
            observation["last_action"] = last_action.copy()
            environment._state = dict(environment._state)
            environment._state["last_action"] = last_action.copy()
    terminated = False
    truncated = False
    trf_mus: list[float] = []
    while not (terminated or truncated):
        if model is None:
            action = np.concatenate([[-1.0], np.zeros(n_actions - 1)]).astype(
                np.float32
            )
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        if not terminated and not truncated:
            trf_mus.append(float(info.get("trf_mu", 1.0)))

    values = np.asarray(environment._asset_memory["final"], dtype=float)
    dates = pd.to_datetime(environment._date_memory)
    target_weights = np.asarray(environment._actions_memory, dtype=float)
    actual_weights = np.asarray(environment._final_weights, dtype=float)
    turnover_by_step = (
        np.abs(target_weights[1:] - actual_weights[:-1]).sum(axis=1)
        if len(target_weights) > 1
        else np.array([], dtype=float)
    )
    steps = max(len(values) - 1, 1)
    turnover = float(turnover_by_step.mean()) if len(turnover_by_step) > 0 else np.nan
    reward_per_day = float(
        kwargs["reward_scaling"] * np.log(values[-1] / values[0]) / steps
    )
    metrics = performance_metrics_from_values(
        values,
        turnover=turnover,
        reward_per_day=reward_per_day,
    )
    return {
        "values": values,
        "dates": dates,
        "target_weights": target_weights,
        "actual_weights": actual_weights,
        "turnover_by_step": turnover_by_step,
        "trf_mu_by_step": np.asarray(trf_mus, dtype=float),
        "weight_labels": ["Cash", *list(environment._tic_list)],
        "reward_per_day": reward_per_day,
        "turnover": turnover,
        "max_weight": (
            float(target_weights[1:].max()) if len(target_weights) > 1 else np.nan
        ),
        "metrics": metrics,
        "final_portfolio_value": float(values[-1]),
        "final_actual_weights": actual_weights[-1].copy(),
        "final_target_weights": target_weights[-1].copy(),
    }


class DualValidationCallback(BaseCallback):
    """Log source diagnostics and select checkpoints on real validation data."""

    def __init__(
        self,
        *,
        train_evaluation: Sequence[pd.DataFrame],
        real_validation: pd.DataFrame,
        synthetic_validation: Sequence[pd.DataFrame],
        kwargs: Mapping[str, Any],
        eval_freq: int,
        best_model_path: Path,
        seed: int,
    ):
        super().__init__()
        self.train_evaluation = list(train_evaluation)
        self.real_validation = real_validation
        self.synthetic_validation = list(synthetic_validation)
        self.kwargs = dict(kwargs)
        self.eval_freq = int(eval_freq)
        self.best_model_path = Path(best_model_path)
        self.seed = int(seed)
        self.records: list[dict[str, Any]] = []
        self.best_score = -np.inf
        self.best_timestep = 0

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        train_results = [
            rollout(frame, self.model, self.kwargs, self.seed)
            for frame in self.train_evaluation
        ]
        train_scores = [result["reward_per_day"] for result in train_results]
        real_result = rollout(self.real_validation, self.model, self.kwargs, self.seed)
        synthetic_scores = [
            rollout(frame, self.model, self.kwargs, self.seed)["reward_per_day"]
            for frame in self.synthetic_validation
        ]
        record = {
            "timesteps": int(self.num_timesteps),
            "train_reward_mean": float(np.mean(train_scores)),
            "train_reward_std": (
                float(np.std(train_scores, ddof=1)) if len(train_scores) > 1 else np.nan
            ),
            "real_valid_reward": float(real_result["reward_per_day"]),
            "real_valid_turnover": float(real_result["turnover"]),
            "real_valid_max_weight": float(real_result["max_weight"]),
            "synthetic_valid_reward_mean": (
                float(np.mean(synthetic_scores)) if synthetic_scores else np.nan
            ),
            "synthetic_valid_reward_std": (
                float(np.std(synthetic_scores, ddof=1))
                if len(synthetic_scores) > 1
                else np.nan
            ),
        }
        record["is_best"] = record["real_valid_reward"] > self.best_score
        if record["is_best"]:
            self.best_score = record["real_valid_reward"]
            self.best_timestep = int(self.num_timesteps)
            self.model.save(self.best_model_path)
        self.records.append(record)


def _training_environment_and_validation(
    *,
    prepared: PreparedWindow,
    config: ExperimentConfig,
    group: str,
    seed: int,
    kwargs: Mapping[str, Any],
) -> tuple[Any, list[pd.DataFrame], pd.DataFrame, list[pd.DataFrame]]:
    model_name = model_from_group(group)
    if group == "real_trained":
        return (
            PortfolioOptimizationGymnasiumEnv(
                prepared.real_pipeline["train"], **dict(kwargs)
            ),
            [prepared.real_pipeline["train"]],
            prepared.real_pipeline["real_valid"],
            [],
        )
    if model_name not in prepared.synthetic_pipelines:
        raise ValueError(f"Unknown training group: {group}")
    pipeline = prepared.synthetic_pipelines[model_name]
    if is_real_synthetic_group(group):
        environment = MultiPathEnv.from_balanced_real_and_synthetic_dataframes(
            pipeline["real_train"],
            pipeline["train_paths"],
            seed=seed,
            **dict(kwargs),
        )
        train_evaluation = [
            pipeline["real_train"],
            *pipeline["train_paths"][: config.max_train_evaluation_paths],
        ]
    else:
        environment = MultiPathEnv.from_dataframes(
            pipeline["train_paths"], seed=seed, **dict(kwargs)
        )
        train_evaluation = pipeline["train_paths"][: config.max_train_evaluation_paths]
    return (
        environment,
        train_evaluation,
        pipeline["real_valid"],
        pipeline["synthetic_valid_paths"][: config.max_synthetic_validation_paths],
    )


def _cached_record(
    plan: TrainingPlan,
    prepared: PreparedWindow,
    config: ExperimentConfig,
) -> dict[str, Any]:
    if plan.cache_match is None:
        raise ValueError("Training plan has no cache match")
    status_path, cached_status = plan.cache_match
    record = dict(cached_status)
    run_dir = status_path.parent
    record.update(
        {
            "cache_hit": True,
            "cache_key": plan.cache_key,
            "cache_schema": TRAIN_CACHE_SCHEMA,
            "cache_dir": str(run_dir),
            "best_model_path": str(run_dir / "best_model"),
            "training_data_fingerprint": prepared.training_data_fingerprints[
                plan.group
            ],
            "training_data_fingerprint_schema": TRAINING_DATA_FINGERPRINT_SCHEMA,
            "semantic_cache_config": plan.cache_config,
            "last_reuse_runtime_provenance": _runtime_provenance(config),
            "window_id": prepared.split.name,
            "ticker_group": prepared.ticker_group,
            "group": plan.group,
            "model_name": model_from_group(plan.group),
            "commission_name": plan.commission_name,
            "commission": plan.commission,
            "seed": plan.seed,
            "run_id": plan.identifier,
        }
    )
    return record


def train_one_run(
    *,
    plan: TrainingPlan,
    prepared: PreparedWindow,
    config: ExperimentConfig,
    settings: RunSettings,
    paths: ExperimentPaths,
    show_progress: bool,
) -> dict[str, Any]:
    if plan.cache_match is not None:
        return _cached_record(plan, prepared, config)

    run_dir = plan.planned_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model"
    last_model_path = run_dir / "last_model"
    set_global_seed(plan.seed)
    kwargs = environment_kwargs(config, plan.commission, cwd=paths.experiment_root)
    training_env, train_evaluation, real_validation, synthetic_validation = (
        _training_environment_and_validation(
            prepared=prepared,
            config=config,
            group=plan.group,
            seed=plan.seed,
            kwargs=kwargs,
        )
    )
    logger = EpisodeLogger()
    validator = DualValidationCallback(
        train_evaluation=train_evaluation,
        real_validation=real_validation,
        synthetic_validation=synthetic_validation,
        kwargs=kwargs,
        eval_freq=settings.eval_freq,
        best_model_path=best_model_path,
        seed=plan.seed,
    )
    callbacks: list[BaseCallback] = [logger, validator]
    if show_progress:
        callbacks.append(
            TqdmTrainingCallback(
                settings.total_timesteps,
                f"{prepared.split.name} | {prepared.ticker_group} | {plan.group} | "
                f"{plan.commission_name} | seed {plan.seed}",
            )
        )
    model = PPO(
        "MultiInputPolicy",
        Monitor(training_env),
        seed=plan.seed,
        **ppo_kwargs(config),
    )
    model.learn(
        total_timesteps=settings.total_timesteps,
        callback=CallbackList(callbacks),
    )
    model.save(last_model_path)
    pd.DataFrame(logger.records).to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame(validator.records).to_csv(run_dir / "validation_log.csv", index=False)
    runtime = _runtime_provenance(config)
    status = {
        "completed": True,
        "cache_hit": False,
        "cache_key": plan.cache_key,
        "cache_schema": TRAIN_CACHE_SCHEMA,
        "cache_dir": str(run_dir),
        "training_data_fingerprint": prepared.training_data_fingerprints[plan.group],
        "training_data_fingerprint_schema": TRAINING_DATA_FINGERPRINT_SCHEMA,
        "semantic_cache_config": plan.cache_config,
        "runtime_provenance": runtime,
        "runtime_fingerprint": _content_hash(runtime),
        "experiment_id": paths.experiment_id,
        "window_id": prepared.split.name,
        "ticker_group": prepared.ticker_group,
        "run_id": plan.identifier,
        "run_mode": settings.mode,
        "group": plan.group,
        "model_name": model_from_group(plan.group),
        "commission_name": plan.commission_name,
        "commission": plan.commission,
        "seed": plan.seed,
        "total_timesteps": settings.total_timesteps,
        "episode_source_sampling": plan.cache_config.get("episode_source_sampling"),
        "best_timestep": validator.best_timestep,
        "best_real_validation_reward": validator.best_score,
        "best_model_path": str(best_model_path),
    }
    (run_dir / "status.json").write_text(
        json.dumps(_serializable(status), indent=2, sort_keys=True) + "\n"
    )
    return status


def run_training_matrix(
    *,
    prepared: PreparedWindow,
    config: ExperimentConfig,
    settings: RunSettings,
    paths: ExperimentPaths,
    use_cache: bool = True,
    force_retrain: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    groups = training_groups(prepared.model_names)
    specs = [
        (group, commission_name, commission, seed)
        for commission_name, commission in settings.commissions
        for seed in settings.seeds
        for group in groups
    ]
    plans = [
        plan_training_run(
            config=config,
            settings=settings,
            prepared=prepared,
            paths=paths,
            group=group,
            commission_name=commission_name,
            commission=commission,
            seed=seed,
            use_cache=use_cache,
            force_retrain=force_retrain,
        )
        for group, commission_name, commission, seed in tqdm(
            specs,
            desc=f"{prepared.split.name}/{prepared.ticker_group}: checking cache",
            unit="run",
            disable=not show_progress,
        )
    ]
    cache_hits = [plan for plan in plans if plan.cache_match is not None]
    cache_misses = [plan for plan in plans if plan.cache_match is None]
    print(
        f"{prepared.split.name}/{prepared.ticker_group}: "
        f"{len(cache_hits)} cache hit(s), "
        f"{len(cache_misses)} training run(s)"
    )
    paths.experiment_root.mkdir(parents=True, exist_ok=True)
    timing_path = paths.experiment_root / "run_timings.csv"
    timing_records: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}

    for plan in tqdm(
        [*cache_hits, *cache_misses],
        desc=f"{prepared.split.name}/{prepared.ticker_group}: training matrix",
        unit="run",
        disable=not show_progress,
    ):
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        status = "failed"
        error_type = ""
        try:
            records_by_id[plan.identifier] = train_one_run(
                plan=plan,
                prepared=prepared,
                config=config,
                settings=settings,
                paths=paths,
                show_progress=show_progress and plan.cache_match is None,
            )
            status = "completed"
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            finished_at = datetime.now(timezone.utc)
            timing_records.append(
                {
                    "execution_order": len(timing_records) + 1,
                    "window_id": prepared.split.name,
                    "ticker_group": prepared.ticker_group,
                    "run_mode": settings.mode,
                    "run_id": plan.identifier,
                    "group": plan.group,
                    "model_name": model_from_group(plan.group),
                    "commission_name": plan.commission_name,
                    "seed": plan.seed,
                    "cache_hit": plan.cache_match is not None,
                    "status": status,
                    "error_type": error_type,
                    "started_at_utc": started_at.isoformat(),
                    "finished_at_utc": finished_at.isoformat(),
                    "elapsed_seconds": round(time.perf_counter() - started_clock, 6),
                }
            )
            pd.DataFrame(timing_records).to_csv(timing_path, index=False)
    return [records_by_id[plan.identifier] for plan in plans]


def _period_frame(
    prepared: PreparedWindow, group: str, model_name: str | None, period: str
) -> pd.DataFrame:
    real_keys = {"train": "train", "validation": "real_valid", "test": "real_test"}
    synthetic_keys = {
        "train": "real_train",
        "validation": "real_valid",
        "test": "real_test",
    }
    if group == "real_trained":
        return prepared.real_pipeline[real_keys[period]]
    if model_name is None:
        raise ValueError(f"Training group {group} has no synthetic model name")
    return prepared.synthetic_pipelines[model_name][synthetic_keys[period]]


def _add_split_columns(
    frame: pd.DataFrame,
    split: TimeSplit,
    ticker_group: str,
    dataset_id: str,
) -> pd.DataFrame:
    enriched = frame.copy()
    columns = {
        "window_id": split.name,
        "ticker_group": ticker_group,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "val_start": split.val_start,
        "val_end": split.val_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "synthetic_dataset_id": dataset_id,
    }
    for position, (name, value) in enumerate(columns.items()):
        if name in enriched.columns:
            enriched = enriched.drop(columns=name)
        enriched.insert(position, name, value)
    return enriched


def evaluate_window(
    *,
    prepared: PreparedWindow,
    config: ExperimentConfig,
    settings: RunSettings,
    paths: ExperimentPaths,
    run_records: Sequence[Mapping[str, Any]],
) -> WindowResult:
    """Evaluate cached/trained policies and persist notebook-ready artifacts."""
    groups = training_groups(prepared.model_names)
    group_labels = {group: training_group_label(config, group) for group in groups}
    ppo_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for record in run_records:
        group = str(record["group"])
        model_name = record.get("model_name")
        commission_name = str(record["commission_name"])
        commission = float(record["commission"])
        seed = int(record["seed"])
        model = PPO.load(record["best_model_path"], device=config.device)
        kwargs = environment_kwargs(config, commission, cwd=paths.experiment_root)
        for period in ("train", "validation", "test"):
            frame = _period_frame(prepared, group, model_name, period)
            result = rollout(frame, model, kwargs, seed)
            ppo_rows.append(
                {
                    "run_id": record["run_id"],
                    "period": period,
                    "evaluation_mode": "independent",
                    "agent": group_labels[group],
                    "agent_type": "PPO",
                    "group": group,
                    "model_name": model_name,
                    "commission_name": commission_name,
                    "commission": commission,
                    "seed": seed,
                    "best_timestep": int(record["best_timestep"]),
                    **result["metrics"],
                }
            )
            for date, value in zip(
                result["dates"], result["values"] / result["values"][0]
            ):
                equity_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "period": period,
                        "evaluation_mode": "independent",
                        "agent": group_labels[group],
                        "agent_type": "PPO",
                        "group": group,
                        "model_name": model_name,
                        "commission_name": commission_name,
                        "seed": seed,
                        "growth_of_one": float(value),
                    }
                )
            if period == "test":
                labels = result["weight_labels"]
                dates = result["dates"][1:]
                actual = result["actual_weights"][1:]
                target = result["target_weights"][1:]
                for date_index, date in enumerate(dates):
                    for asset_index, asset in enumerate(labels):
                        weight_rows.append(
                            {
                                "date": pd.Timestamp(date),
                                "evaluation_mode": "independent",
                                "group": group,
                                "model_name": model_name,
                                "commission_name": commission_name,
                                "seed": seed,
                                "asset": asset,
                                "target_weight": float(target[date_index, asset_index]),
                                "actual_weight": float(actual[date_index, asset_index]),
                            }
                        )

    for period in ("train", "validation", "test"):
        for commission_name, commission in settings.commissions:
            kwargs = environment_kwargs(config, commission, cwd=paths.experiment_root)
            real_key = {
                "train": "train",
                "validation": "real_valid",
                "test": "real_test",
            }[period]
            result = rollout(
                prepared.real_pipeline[real_key],
                None,
                kwargs,
                settings.seeds[0],
            )
            benchmark_rows.append(
                {
                    "run_id": f"buy_and_hold__{period}__{commission_name}",
                    "period": period,
                    "evaluation_mode": "independent",
                    "agent": "Buy & Hold",
                    "agent_type": "Buy & Hold",
                    "group": "buy_and_hold",
                    "model_name": None,
                    "commission_name": commission_name,
                    "commission": commission,
                    "seed": pd.NA,
                    "best_timestep": np.nan,
                    **result["metrics"],
                }
            )
            for date, value in zip(
                result["dates"], result["values"] / result["values"][0]
            ):
                equity_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "period": period,
                        "evaluation_mode": "independent",
                        "agent": "Buy & Hold",
                        "agent_type": "Buy & Hold",
                        "group": "buy_and_hold",
                        "model_name": None,
                        "commission_name": commission_name,
                        "seed": pd.NA,
                        "growth_of_one": float(value),
                    }
                )

    ppo_period = _add_split_columns(
        pd.DataFrame(ppo_rows),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )
    test_metrics = ppo_period.loc[ppo_period["period"] == "test"].reset_index(drop=True)
    test_days = len(
        _execution_dates(
            prepared.real_pipeline["real_test"],
            prepared.split.test_start,
            prepared.split.test_end,
        )
    )
    test_metrics["test_days"] = test_days
    test_metrics["is_partial_window"] = (
        config.rolling_schedule is not None
        and test_days < config.rolling_schedule.test_days
    )
    benchmark = _add_split_columns(
        pd.DataFrame(benchmark_rows),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )
    backtest = pd.concat([ppo_period, benchmark], ignore_index=True)
    equity = _add_split_columns(
        pd.DataFrame(equity_rows),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )
    weights = _add_split_columns(
        pd.DataFrame(weight_rows),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )

    metric_columns = [
        "cumulative_return",
        "cagr",
        "sharpe",
        "annualized_volatility",
        "max_drawdown",
        "turnover",
        "best_timestep",
    ]
    aggregate = test_metrics.groupby(
        ["window_id", "ticker_group", "commission_name", "group"]
    )[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["agent"] = aggregate["group"].map(group_labels)

    paired_rows: list[dict[str, Any]] = []
    for commission_name, _ in settings.commissions:
        fee_rows = test_metrics.loc[test_metrics["commission_name"] == commission_name]
        for seed in settings.seeds:
            pair = fee_rows.loc[fee_rows["seed"] == seed].set_index("group")
            if "real_trained" not in pair.index:
                continue
            for model_name in prepared.model_names:
                for group in (
                    synthetic_group(model_name),
                    real_synthetic_group(model_name),
                ):
                    if group not in pair.index:
                        continue
                    row = {
                        "commission_name": commission_name,
                        "seed": seed,
                        "model_name": model_name,
                        "model_label": synthetic_model_label(config, model_name),
                        "training_group": group,
                        "training_group_label": group_labels[group],
                    }
                    for metric in metric_columns:
                        row[f"delta_{metric}"] = float(
                            pair.loc[group, metric] - pair.loc["real_trained", metric]
                        )
                    paired_rows.append(row)
    paired = _add_split_columns(
        pd.DataFrame(paired_rows),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )

    representatives = []
    for (_, group), subset in test_metrics.groupby(
        ["commission_name", "group"], sort=False
    ):
        candidates = subset.dropna(subset=["sharpe"]).copy()
        if candidates.empty:
            ordered = subset.sort_values("seed").reset_index(drop=True)
            selected = ordered.iloc[len(ordered) // 2].copy()
            selected["median_test_sharpe"] = np.nan
            selected["sharpe_distance_to_median"] = np.nan
        else:
            median = float(candidates["sharpe"].median())
            candidates["_distance"] = (candidates["sharpe"] - median).abs()
            selected = candidates.sort_values(["_distance", "seed"]).iloc[0].copy()
            selected["median_test_sharpe"] = median
            selected["sharpe_distance_to_median"] = float(selected["_distance"])
            selected = selected.drop(labels=["_distance"])
        representatives.append(selected)
    representative_runs = pd.DataFrame(representatives).reset_index(drop=True)

    paths.experiment_root.mkdir(parents=True, exist_ok=True)
    run_table = _add_split_columns(
        pd.DataFrame(run_records),
        prepared.split,
        prepared.ticker_group,
        prepared.dataset_id,
    )
    artifacts = {
        "run_table.csv": run_table,
        "data_summary.csv": prepared.data_summary,
        "scale_summary.csv": prepared.scale_summary,
        "ppo_period_metrics.csv": ppo_period,
        "test_metrics.csv": test_metrics,
        "buy_and_hold_metrics.csv": benchmark,
        "backtest_results.csv": backtest,
        "equity_curves.csv": equity,
        "portfolio_weights.csv": weights,
        "aggregate_summary.csv": aggregate,
        "paired_deltas.csv": paired,
        "median_sharpe_representatives.csv": representative_runs,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(paths.experiment_root / filename, index=False)
    model_references = [
        {
            "window_id": prepared.split.name,
            "ticker_group": prepared.ticker_group,
            "run_id": record["run_id"],
            "group": record["group"],
            "commission_name": record["commission_name"],
            "seed": int(record["seed"]),
            "best_model_path": record["best_model_path"],
        }
        for record in run_records
    ]
    (paths.experiment_root / "model_references.json").write_text(
        json.dumps(_serializable(model_references), indent=2, sort_keys=True) + "\n"
    )
    return WindowResult(
        split=prepared.split,
        ticker_group=prepared.ticker_group,
        dataset_id=prepared.dataset_id,
        tickers=prepared.tickers,
        experiment_root=paths.experiment_root,
        experiment_id=paths.experiment_id,
        run_table=run_table,
        test_metrics=test_metrics,
        aggregate_summary=aggregate,
        paired_deltas=paired,
        representative_runs=representative_runs,
        prepared=prepared,
        run_records=[dict(record) for record in run_records],
    )


def evaluate_continuous_test_chains(
    *,
    window_results: Sequence[WindowResult],
    config: ExperimentConfig,
    settings: RunSettings,
) -> dict[str, pd.DataFrame]:
    """Evaluate ordered test windows while carrying portfolio state forward."""
    metric_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    by_ticker_group: dict[str, list[WindowResult]] = {}
    for result in window_results:
        by_ticker_group.setdefault(result.ticker_group, []).append(result)

    for ticker_group, unsorted_results in by_ticker_group.items():
        ordered_results = sorted(
            unsorted_results, key=lambda result: pd.Timestamp(result.split.test_start)
        )
        for previous, current in zip(ordered_results, ordered_results[1:]):
            if previous.split.test_end != current.split.test_start:
                raise ValueError(
                    "Continuous evaluation requires contiguous test windows; "
                    f"{previous.split.name} ends {previous.split.test_end}, but "
                    f"{current.split.name} starts {current.split.test_start}"
                )

        chain_states: dict[tuple[str, str, int], dict[str, Any]] = {}
        chain_values: dict[tuple[str, str, int], list[float]] = {}
        chain_turnover: dict[tuple[str, str, int], list[float]] = {}
        chain_metadata: dict[tuple[str, str, int], dict[str, Any]] = {}

        for window_index, window in enumerate(ordered_results):
            prepared = window.prepared
            for record in window.run_records:
                group = str(record["group"])
                model_name = record.get("model_name")
                commission_name = str(record["commission_name"])
                commission = float(record["commission"])
                seed = int(record["seed"])
                key = (group, commission_name, seed)
                cash_weights = np.zeros(len(prepared.tickers) + 1, dtype=float)
                cash_weights[0] = 1.0
                state = chain_states.get(
                    key,
                    {
                        "portfolio_value": float(config.initial_amount),
                        "actual_weights": cash_weights,
                        "last_action": cash_weights,
                    },
                )
                initial_value = float(state["portfolio_value"])
                initial_weights = np.asarray(state["actual_weights"], dtype=float)
                model = PPO.load(record["best_model_path"], device=config.device)
                kwargs = environment_kwargs(
                    config, commission, cwd=window.experiment_root
                )
                frame = _period_frame(prepared, group, model_name, "test")
                rollout_result = rollout(
                    frame,
                    model,
                    kwargs,
                    seed,
                    initial_portfolio_value=initial_value,
                    initial_actual_weights=initial_weights,
                    initial_last_action=state["last_action"],
                )
                values = np.asarray(rollout_result["values"], dtype=float)
                execution_dates = pd.to_datetime(rollout_result["dates"][1:])
                daily_returns = np.diff(values) / values[:-1]
                turnovers = np.asarray(rollout_result["turnover_by_step"], dtype=float)
                trf_mus = np.asarray(rollout_result["trf_mu_by_step"], dtype=float)
                if not (len(execution_dates) == len(daily_returns) == len(turnovers)):
                    raise RuntimeError("Continuous rollout arrays are misaligned")
                group_label = training_group_label(config, group)
                window_metric_rows.append(
                    {
                        "window_id": window.split.name,
                        "ticker_group": ticker_group,
                        "synthetic_dataset_id": window.dataset_id,
                        "test_start": window.split.test_start,
                        "test_end": window.split.test_end,
                        "evaluation_mode": "continuous",
                        "agent": group_label,
                        "group": group,
                        "model_name": model_name,
                        "commission_name": commission_name,
                        "commission": commission,
                        "seed": seed,
                        "best_timestep": int(record["best_timestep"]),
                        "test_days": len(daily_returns),
                        "is_partial_window": (
                            config.rolling_schedule is not None
                            and len(daily_returns) < config.rolling_schedule.test_days
                        ),
                        **rollout_result["metrics"],
                    }
                )
                transition_rows.append(
                    {
                        "window_index": window_index,
                        "window_id": window.split.name,
                        "ticker_group": ticker_group,
                        "group": group,
                        "commission_name": commission_name,
                        "seed": seed,
                        "initial_portfolio_value": initial_value,
                        "final_portfolio_value": rollout_result[
                            "final_portfolio_value"
                        ],
                        "initial_actual_weights": json.dumps(
                            initial_weights.tolist(), separators=(",", ":")
                        ),
                        "first_target_weights": json.dumps(
                            rollout_result["target_weights"][1].tolist(),
                            separators=(",", ":"),
                        ),
                        "final_actual_weights": json.dumps(
                            rollout_result["final_actual_weights"].tolist(),
                            separators=(",", ":"),
                        ),
                        "final_target_weights": json.dumps(
                            rollout_result["final_target_weights"].tolist(),
                            separators=(",", ":"),
                        ),
                        "boundary_turnover": (
                            float(turnovers[0]) if len(turnovers) else np.nan
                        ),
                        "boundary_fee": (
                            initial_value * (1.0 - float(trf_mus[0]))
                            if len(trf_mus)
                            else 0.0
                        ),
                    }
                )
                labels = rollout_result["weight_labels"]
                targets = rollout_result["target_weights"][1:]
                actuals = rollout_result["actual_weights"][1:]
                for step_index, (date, daily_return, value, turnover) in enumerate(
                    zip(execution_dates, daily_returns, values[1:], turnovers)
                ):
                    common = {
                        "date": date,
                        "window_id": window.split.name,
                        "ticker_group": ticker_group,
                        "synthetic_dataset_id": window.dataset_id,
                        "evaluation_mode": "continuous",
                        "agent": group_label,
                        "group": group,
                        "model_name": model_name,
                        "commission_name": commission_name,
                        "commission": commission,
                        "seed": seed,
                    }
                    daily_rows.append(
                        {
                            **common,
                            "daily_return": float(daily_return),
                            "portfolio_value": float(value),
                            "turnover": float(turnover),
                        }
                    )
                    equity_rows.append(
                        {
                            **common,
                            "growth_of_one": float(value / config.initial_amount),
                        }
                    )
                    for asset_index, asset in enumerate(labels):
                        weight_rows.append(
                            {
                                **common,
                                "asset": asset,
                                "target_weight": float(
                                    targets[step_index, asset_index]
                                ),
                                "actual_weight": float(
                                    actuals[step_index, asset_index]
                                ),
                            }
                        )
                chain_states[key] = {
                    "portfolio_value": rollout_result["final_portfolio_value"],
                    "actual_weights": rollout_result["final_actual_weights"],
                    "last_action": rollout_result["final_target_weights"],
                }
                chain_values.setdefault(key, [float(config.initial_amount)]).extend(
                    values[1:].tolist()
                )
                chain_turnover.setdefault(key, []).extend(turnovers.tolist())
                chain_metadata[key] = {
                    "ticker_group": ticker_group,
                    "synthetic_dataset_id": window.dataset_id,
                    "evaluation_mode": "continuous",
                    "agent": group_label,
                    "group": group,
                    "model_name": model_name,
                    "commission_name": commission_name,
                    "commission": commission,
                    "seed": seed,
                }

        for key, values in chain_values.items():
            turnovers = chain_turnover[key]
            steps = len(values) - 1
            reward_per_day = float(
                config.reward_scaling * np.log(values[-1] / values[0]) / steps
            )
            metric_rows.append(
                {
                    **chain_metadata[key],
                    "test_days": steps,
                    **performance_metrics_from_values(
                        values,
                        turnover=float(np.mean(turnovers)),
                        reward_per_day=reward_per_day,
                    ),
                }
            )

    return {
        "continuous_metrics": pd.DataFrame(metric_rows),
        "continuous_window_metrics": pd.DataFrame(window_metric_rows),
        "continuous_daily_returns": pd.DataFrame(daily_rows),
        "continuous_equity_curves": pd.DataFrame(equity_rows),
        "continuous_portfolio_weights": pd.DataFrame(weight_rows),
        "window_transition_log": pd.DataFrame(transition_rows),
    }


def run_window(
    *,
    project_root: Path,
    config: ExperimentConfig,
    settings: RunSettings,
    split: TimeSplit,
    ticker_group: str,
    tickers: Sequence[str],
    real_files: Sequence[Path],
    real_raw: pd.DataFrame,
    use_cache: bool = True,
    force_retrain: bool = False,
    show_progress: bool = True,
) -> WindowResult:
    prepared = prepare_window_data(
        project_root=project_root,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=real_raw,
        ticker_group=ticker_group,
        tickers=tickers,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        config=config,
        settings=settings,
        prepared=prepared,
    )
    paths.experiment_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": paths.experiment_id,
        "split": asdict(split),
        "ticker_group": ticker_group,
        "tickers": list(tickers),
        "synthetic_dataset_id": prepared.dataset_id,
        "config": config.to_mapping(),
        "run_settings": asdict(settings),
        "data_fingerprint": prepared.data_fingerprint,
        "training_data_fingerprints": prepared.training_data_fingerprints,
    }
    (paths.experiment_root / "experiment_config.json").write_text(
        json.dumps(_serializable(metadata), indent=2, sort_keys=True) + "\n"
    )
    prepared.data_summary.to_csv(
        paths.experiment_root / "data_summary.csv", index=False
    )
    prepared.scale_summary.to_csv(
        paths.experiment_root / "scale_summary.csv", index=False
    )
    run_records = run_training_matrix(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        use_cache=use_cache,
        force_retrain=force_retrain,
        show_progress=show_progress,
    )
    return evaluate_window(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        run_records=run_records,
    )


def _combine_frames(results: Sequence[WindowResult], attribute: str) -> pd.DataFrame:
    frames = [getattr(result, attribute) for result in results]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_experiment_suite(
    *,
    project_root: Path,
    config: ExperimentConfig,
    mode: str,
    split_names: Sequence[str] | None = None,
    ticker_group_names: Sequence[str] | None = None,
    use_cache: bool = True,
    force_retrain: bool = False,
    show_progress: bool = True,
) -> ExperimentSuiteResult:
    """Run selected time-window × ticker-group cases and write suite artifacts."""
    project_root = Path(project_root).resolve()
    settings = config.resolve_run_settings(mode)
    ticker_groups = config.select_ticker_groups(ticker_group_names)
    real_data = {
        ticker_group: load_real_market_data(project_root, config, tickers)
        for ticker_group, tickers in ticker_groups
    }
    results: list[WindowResult] = []
    for ticker_group, tickers in ticker_groups:
        splits = rolling_time_splits(
            project_root=project_root,
            config=config,
            ticker_group=ticker_group,
            tickers=tickers,
            names=split_names,
        )
        for split in splits:
            results.append(
                run_window(
                    project_root=project_root,
                    config=config,
                    settings=settings,
                    split=split,
                    ticker_group=ticker_group,
                    tickers=tickers,
                    real_files=real_data[ticker_group][0],
                    real_raw=real_data[ticker_group][1],
                    use_cache=use_cache,
                    force_retrain=force_retrain,
                    show_progress=show_progress,
                )
            )
    suite_payload = {
        "mode": settings.mode,
        "windows": [
            {
                "case_id": f"{result.split.name}::{result.ticker_group}",
                "window_id": result.split.name,
                "ticker_group": result.ticker_group,
                "synthetic_dataset_id": result.dataset_id,
                "tickers": list(result.tickers),
                "experiment_id": result.experiment_id,
                "experiment_root": str(result.experiment_root),
            }
            for result in results
        ],
        "config": config.to_mapping(),
    }
    suite_id = _content_hash(suite_payload)[:12]
    artifact_root = resolve_project_path(project_root, config.artifact_dir)
    suite_root = artifact_root / settings.mode / "suites" / suite_id
    suite_root.mkdir(parents=True, exist_ok=True)
    test_metrics = _combine_frames(results, "test_metrics")
    aggregate = _combine_frames(results, "aggregate_summary")
    paired = _combine_frames(results, "paired_deltas")
    continuous_artifacts = (
        evaluate_continuous_test_chains(
            window_results=results,
            config=config,
            settings=settings,
        )
        if "continuous" in config.evaluation_modes
        else {
            "continuous_metrics": pd.DataFrame(),
            "continuous_window_metrics": pd.DataFrame(),
            "continuous_daily_returns": pd.DataFrame(),
            "continuous_equity_curves": pd.DataFrame(),
            "continuous_portfolio_weights": pd.DataFrame(),
            "window_transition_log": pd.DataFrame(),
        }
    )
    test_metrics.to_csv(suite_root / "test_metrics.csv", index=False)
    aggregate.to_csv(suite_root / "aggregate_summary.csv", index=False)
    paired.to_csv(suite_root / "paired_deltas.csv", index=False)
    for name, frame in continuous_artifacts.items():
        frame.to_csv(suite_root / f"{name}.csv", index=False)
    manifest = {
        "suite_id": suite_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **suite_payload,
        "combined_artifacts": {
            "test_metrics": str(suite_root / "test_metrics.csv"),
            "aggregate_summary": str(suite_root / "aggregate_summary.csv"),
            "paired_deltas": str(suite_root / "paired_deltas.csv"),
            **{name: str(suite_root / f"{name}.csv") for name in continuous_artifacts},
        },
    }
    manifest_path = suite_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_serializable(manifest), indent=2, sort_keys=True) + "\n"
    )
    latest_path = artifact_root / settings.mode / "latest_suite.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {"manifest_path": str(manifest_path), "suite_id": suite_id},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return ExperimentSuiteResult(
        manifest_path=manifest_path,
        suite_root=suite_root,
        window_results=results,
        test_metrics=test_metrics,
        aggregate_summary=aggregate,
        paired_deltas=paired,
        continuous_metrics=continuous_artifacts["continuous_metrics"],
    )
