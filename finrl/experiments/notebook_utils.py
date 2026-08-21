from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm.auto import tqdm


NORMALIZED_OHLCV_COLUMNS = (
    "date",
    "tic",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
SOURCE_OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def environment_flag(name: str, default: bool) -> bool:
    """Read a strict boolean environment flag shared by experiment notebooks."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw_value!r}")


def find_project_root(start: Path) -> Path:
    """Find the nearest parent containing both the FinRL package and examples."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "finrl").is_dir() and (candidate / "examples").is_dir():
            return candidate
    raise RuntimeError("Could not locate the FinRL project root.")


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_periods(periods: Sequence[tuple[str, str, str]]) -> None:
    """Validate chronological, non-overlapping half-open date periods."""
    previous_end: pd.Timestamp | None = None
    for label, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts >= end_ts:
            raise ValueError(f"{label}: start must be before end")
        if previous_end is not None and previous_end > start_ts:
            raise ValueError(f"{label}: period overlaps the previous period")
        previous_end = end_ts


def validate_long_market_frame(
    frame: pd.DataFrame,
    label: str,
    expected_tickers: Sequence[str],
    required_columns: Sequence[str],
    positive_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate and sort a FinRL long-format market frame."""
    required = list(required_columns)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing normalized columns {missing}")
    validated = frame[required].copy()
    validated["date"] = pd.to_datetime(validated["date"], errors="raise").dt.normalize()
    validated["tic"] = validated["tic"].astype(str).str.upper()

    numeric_columns = [column for column in required if column not in {"date", "tic"}]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="raise")

    expected = sorted(str(ticker).upper() for ticker in expected_tickers)
    actual = sorted(validated["tic"].unique())
    if actual != expected:
        raise ValueError(f"{label}: expected tickers {expected}, got {actual}")
    if validated.duplicated(["date", "tic"]).any():
        examples = validated.loc[
            validated.duplicated(["date", "tic"], keep=False), ["date", "tic"]
        ].head()
        raise ValueError(f"{label}: duplicate date/tic rows:\n{examples}")

    numbers = validated[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError(f"{label}: numeric columns contain NaN or infinity")
    if (validated[list(positive_columns)] <= 0).any().any():
        raise ValueError(f"{label}: positive columns must be strictly positive")
    if "volume" in validated and (validated["volume"] < 0).any():
        raise ValueError(f"{label}: volume must be non-negative")

    if {"low", "close", "high"}.issubset(validated.columns):
        invalid = (validated["low"] > validated["close"]) | (
            validated["close"] > validated["high"]
        )
        if invalid.any():
            raise ValueError(
                f"{label}: {int(invalid.sum())} rows violate low <= close <= high"
            )

    date_sets = validated.groupby("tic", sort=True)["date"].apply(
        lambda values: frozenset(values)
    )
    if date_sets.nunique() != 1:
        counts = validated.groupby("tic")["date"].nunique().to_dict()
        raise ValueError(f"{label}: ticker date sets differ: {counts}")
    return validated.sort_values(["date", "tic"], ignore_index=True)


def read_long_market_csv(
    path: Path,
    label: str,
    expected_tickers: Sequence[str],
    required_columns: Sequence[str],
    positive_columns: Sequence[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: CSV not found: {path}")
    return validate_long_market_frame(
        pd.read_csv(path),
        label,
        expected_tickers,
        required_columns,
        positive_columns,
    )


def read_real_ohlcv_csvs(
    paths: Iterable[Path], expected_tickers: Sequence[str]
) -> pd.DataFrame:
    """Read adjusted per-ticker OHLCV CSVs into FinRL long format."""
    path_list = list(paths)
    if not path_list:
        raise FileNotFoundError("No per-ticker real CSV files were provided")
    expected = sorted(str(ticker).upper() for ticker in expected_tickers)
    actual = sorted(path.stem.upper() for path in path_list)
    if len(actual) != len(set(actual)):
        raise ValueError(f"Duplicate real-data ticker files: {actual}")
    if actual != expected:
        raise ValueError(f"Real-data filenames must identify {expected}, got {actual}")

    frames = []
    for path in sorted(path_list):
        ticker = path.stem.upper()
        raw = pd.read_csv(path)
        missing = sorted(set(SOURCE_OHLCV_COLUMNS) - set(raw.columns))
        if missing:
            raise ValueError(f"real {ticker}: missing columns {missing}")
        raw = raw[list(SOURCE_OHLCV_COLUMNS)].copy()
        raw["Date"] = pd.to_datetime(raw["Date"], errors="raise").dt.normalize()
        for column in SOURCE_OHLCV_COLUMNS[1:]:
            raw[column] = pd.to_numeric(raw[column], errors="raise")

        values = raw[list(SOURCE_OHLCV_COLUMNS[1:])].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"real {ticker}: OHLCV contains NaN or infinity")
        if (raw[["Open", "High", "Low", "Close"]] <= 0).any().any():
            raise ValueError(f"real {ticker}: adjusted OHLC must be positive")
        if (raw["Volume"] < 0).any():
            raise ValueError(f"real {ticker}: volume must be non-negative")

        row_scale = raw[["Open", "High", "Low", "Close"]].max(axis=1).clip(lower=1)
        tolerance = row_scale * 1e-10
        lower_body = raw[["Open", "Close"]].min(axis=1)
        upper_body = raw[["Open", "Close"]].max(axis=1)
        invalid = (raw["Low"] > lower_body + tolerance) | (
            raw["High"] < upper_body - tolerance
        )
        if invalid.any():
            raise ValueError(
                f"real {ticker}: {int(invalid.sum())} bars violate OHLC ordering"
            )

        raw["Low"] = raw[["Low", "Open", "Close"]].min(axis=1)
        raw["High"] = raw[["High", "Open", "Close"]].max(axis=1)
        converted = raw.rename(columns=str.lower)
        converted["tic"] = ticker
        frames.append(converted[list(NORMALIZED_OHLCV_COLUMNS)])

    combined = pd.concat(frames, ignore_index=True)
    return validate_long_market_frame(
        combined,
        "real data",
        expected,
        NORMALIZED_OHLCV_COLUMNS,
        ("open", "high", "low", "close"),
    )


def check_calendar_coverage(
    frame: pd.DataFrame,
    start: str,
    end: str,
    label: str,
    tolerance_days: int = 7,
) -> None:
    dates = pd.Index(pd.to_datetime(frame["date"].unique())).sort_values()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = dates[(dates >= start_ts) & (dates < end_ts)]
    if selected.empty:
        raise ValueError(f"{label}: no dates in [{start}, {end})")
    tolerance = pd.Timedelta(days=tolerance_days)
    if selected.min() > start_ts + tolerance:
        raise ValueError(
            f"{label}: first date {selected.min().date()} is too far after {start}"
        )
    if selected.max() < end_ts - tolerance:
        raise ValueError(
            f"{label}: last date {selected.max().date()} is too far before {end}"
        )


def slice_period(
    frame: pd.DataFrame,
    start: str,
    end: str,
    label: str,
    minimum_dates: int = 2,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    selected = frame[
        (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    ].copy()
    date_count = selected["date"].nunique()
    if date_count < minimum_dates:
        raise ValueError(
            f"{label}: {date_count} dates are fewer than required {minimum_dates}"
        )
    return selected.sort_values(["date", "tic"], ignore_index=True)


def slice_evaluation_with_lookback(
    frame: pd.DataFrame,
    start: str,
    end: str,
    label: str,
    time_window: int,
) -> pd.DataFrame:
    all_dates = pd.Index(pd.to_datetime(frame["date"].unique())).sort_values()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    period_dates = all_dates[(all_dates >= start_ts) & (all_dates < end_ts)]
    lookback_dates = all_dates[all_dates < start_ts][-time_window:]
    if len(lookback_dates) != time_window:
        raise ValueError(
            f"{label}: needs {time_window} dates before {start}, "
            f"found {len(lookback_dates)}"
        )
    if len(period_dates) < 2:
        raise ValueError(f"{label}: evaluation period has fewer than two dates")
    selected_dates = lookback_dates.append(period_dates)
    return (
        frame[frame["date"].isin(selected_dates)]
        .copy()
        .sort_values(["date", "tic"], ignore_index=True)
    )


class PriceScaleByTicker:
    def __init__(self, tickers: Sequence[str], columns: Sequence[str]):
        self.tickers = sorted(str(ticker).upper() for ticker in tickers)
        self.columns = list(columns)
        self.scales: dict[str, float] = {}

    def fit(self, frames: Sequence[pd.DataFrame]) -> "PriceScaleByTicker":
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("Scaler requires at least one frame")
        actual_tickers: set[str] = set()
        ticker_indexes = {ticker: index for index, ticker in enumerate(self.tickers)}
        maxima = np.full(len(self.tickers), -np.inf, dtype=float)
        for frame in frame_list:
            frame_tickers = frame["tic"].astype(str)
            actual_tickers.update(frame_tickers.unique())
            indexes = frame_tickers.map(ticker_indexes)
            known = indexes.notna().to_numpy()
            if not known.any():
                continue
            values = np.abs(frame[self.columns].to_numpy(dtype=float))
            row_maxima = values.max(axis=1)
            np.maximum.at(
                maxima,
                indexes.to_numpy(dtype=float)[known].astype(np.intp),
                row_maxima[known],
            )
        actual = sorted(actual_tickers)
        if actual != self.tickers:
            raise ValueError(f"Scaler expected {self.tickers}, got {actual}")
        for ticker, scale_value in zip(self.tickers, maxima):
            scale = float(scale_value)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid scale for {ticker}: {scale}")
            self.scales[ticker] = scale
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.scales:
            raise RuntimeError("Scaler must be fit before transform")
        transformed = frame.copy()
        divisors = transformed["tic"].map(self.scales)
        known = divisors.notna().to_numpy()
        values = transformed[self.columns].to_numpy(dtype=float, copy=True)
        values[known] /= divisors.to_numpy(dtype=float)[known, None]
        transformed[self.columns] = values
        return transformed


class EIIEExtractor(BaseFeaturesExtractor):
    """EIIE convolution stack shared by portfolio PPO experiments."""

    def __init__(
        self,
        observation_space,
        k_size: int = 3,
        conv_mid: int = 2,
        conv_final: int = 20,
    ):
        n_features, n_assets, time_window = observation_space["state"].shape
        n_last = observation_space["last_action"].shape[0]
        super().__init__(
            observation_space,
            features_dim=conv_final * n_assets + n_last,
        )
        final_kernel = time_window - k_size + 1
        self.conv = nn.Sequential(
            nn.Conv2d(n_features, conv_mid, kernel_size=(1, k_size)),
            nn.ReLU(),
            nn.Conv2d(conv_mid, conv_final, kernel_size=(1, final_kernel)),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations):
        return torch.cat(
            [self.conv(observations["state"]), observations["last_action"]],
            dim=1,
        )


class EpisodeLogger(BaseCallback):
    """Collect episode reward and selected PPO diagnostics."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.records.append(
                    {
                        "timesteps": self.num_timesteps,
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                    }
                )
        return True

    def _on_rollout_end(self) -> None:
        if not self.records:
            return
        values = self.model.logger.name_to_value
        self.records[-1].update(
            {
                key: values.get(f"train/{key}")
                for key in ("explained_variance", "entropy_loss", "value_loss")
            }
        )


class TqdmTrainingCallback(BaseCallback):
    """Render one live progress bar for a training run."""

    def __init__(self, total_timesteps: int, description: str):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.description = description
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc=self.description,
            unit="step",
            leave=False,
        )

    def _on_step(self) -> bool:
        completed = min(int(self.num_timesteps), self.total_timesteps)
        self.progress_bar.update(completed - self.progress_bar.n)
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update(self.total_timesteps - self.progress_bar.n)
            self.progress_bar.close()


def performance_metrics_from_values(
    values: Sequence[float],
    turnover: float = np.nan,
    reward_per_day: float = np.nan,
    annualization: int = 252,
) -> dict[str, float]:
    asset_values = np.asarray(values, dtype=float)
    if asset_values.ndim != 1 or len(asset_values) < 2:
        raise ValueError("values must be a one-dimensional series of length >= 2")
    if not np.isfinite(asset_values).all() or (asset_values <= 0).any():
        raise ValueError("values must be finite and strictly positive")
    returns = np.diff(asset_values) / asset_values[:-1]
    peak = np.maximum.accumulate(asset_values)
    n_steps = len(returns)
    total_return = asset_values[-1] / asset_values[0] - 1.0
    return_std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    volatility = (
        float(return_std * np.sqrt(annualization))
        if np.isfinite(return_std)
        else np.nan
    )
    sharpe = (
        float(returns.mean() / return_std * np.sqrt(annualization))
        if np.isfinite(return_std) and return_std > 0
        else np.nan
    )
    return {
        "final_value": float(asset_values[-1]),
        "cumulative_return": float(total_return),
        "cagr": float(
            (asset_values[-1] / asset_values[0]) ** (annualization / n_steps) - 1
        ),
        "sharpe": sharpe,
        "annualized_volatility": volatility,
        "max_drawdown": float(((asset_values - peak) / peak).min()),
        "turnover": float(turnover),
        "reward_per_day": float(reward_per_day),
    }
