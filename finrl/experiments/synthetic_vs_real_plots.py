"""Load and visualize artifacts from Synthetic-vs-Real experiment suites."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SuiteArtifacts:
    manifest_path: Path
    manifest: dict[str, Any]
    test_metrics: pd.DataFrame
    aggregate_summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    continuous_metrics: pd.DataFrame
    continuous_window_metrics: pd.DataFrame
    continuous_daily_returns: pd.DataFrame
    continuous_equity_curves: pd.DataFrame
    continuous_portfolio_weights: pd.DataFrame
    window_transition_log: pd.DataFrame
    case_roots: dict[tuple[str, str], Path]

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(window_id for window_id, _ in self.case_roots))

    @property
    def ticker_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(group for _, group in self.case_roots))

    @property
    def experiment_cases(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.case_roots)

    def select_ticker_group(self, ticker_group: str | None) -> str:
        if ticker_group is not None:
            if ticker_group not in self.ticker_groups:
                raise KeyError(f"Unknown ticker group {ticker_group!r}")
            return ticker_group
        if len(self.ticker_groups) != 1:
            raise ValueError(
                "Suite contains multiple ticker groups; pass ticker_group explicitly"
            )
        return self.ticker_groups[0]

    def read_window_csv(
        self,
        window_id: str,
        filename: str,
        *,
        ticker_group: str | None = None,
    ) -> pd.DataFrame:
        selected_group = self.select_ticker_group(ticker_group)
        case = (window_id, selected_group)
        if case not in self.case_roots:
            raise KeyError(
                f"Unknown experiment case window={window_id!r}, "
                f"ticker_group={selected_group!r}"
            )
        path = self.case_roots[case] / filename
        if not path.is_file():
            raise FileNotFoundError(f"Suite artifact not found: {path}")
        return pd.read_csv(path)


def resolve_manifest_path(path: str | Path) -> Path:
    """Resolve either a suite manifest or a ``latest_suite.json`` pointer."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Suite manifest not found: {candidate}")
    with candidate.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    referenced = payload.get("manifest_path")
    if referenced:
        target = Path(referenced).expanduser()
        if not target.is_absolute():
            target = candidate.parent / target
        target = target.resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Referenced suite manifest not found: {target}")
        return target
    if "windows" not in payload or "combined_artifacts" not in payload:
        raise ValueError(f"File is not a Synthetic-vs-Real suite manifest: {candidate}")
    return candidate


def load_suite_artifacts(path: str | Path) -> SuiteArtifacts:
    manifest_path = resolve_manifest_path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    combined = manifest["combined_artifacts"]

    def read_combined(name: str, *, required: bool = True) -> pd.DataFrame:
        if name not in combined:
            if required:
                raise KeyError(f"Suite manifest has no combined artifact {name!r}")
            return pd.DataFrame()
        artifact_path = Path(combined[name]).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        return pd.read_csv(artifact_path.resolve())

    case_roots = {
        (str(row["window_id"]), str(row.get("ticker_group", "default"))): Path(
            row["experiment_root"]
        )
        .expanduser()
        .resolve()
        for row in manifest["windows"]
    }
    test_metrics = read_combined("test_metrics")
    aggregate_summary = read_combined("aggregate_summary")
    paired_deltas = read_combined("paired_deltas")
    for frame in (test_metrics, aggregate_summary, paired_deltas):
        if "ticker_group" not in frame.columns:
            frame.insert(1, "ticker_group", "default")
    return SuiteArtifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        test_metrics=test_metrics,
        aggregate_summary=aggregate_summary,
        paired_deltas=paired_deltas,
        continuous_metrics=read_combined("continuous_metrics", required=False),
        continuous_window_metrics=read_combined(
            "continuous_window_metrics", required=False
        ),
        continuous_daily_returns=read_combined(
            "continuous_daily_returns", required=False
        ),
        continuous_equity_curves=read_combined(
            "continuous_equity_curves", required=False
        ),
        continuous_portfolio_weights=read_combined(
            "continuous_portfolio_weights", required=False
        ),
        window_transition_log=read_combined("window_transition_log", required=False),
        case_roots=case_roots,
    )


def _default_group_labels(groups: Sequence[str]) -> dict[str, str]:
    labels = {}
    for group in groups:
        if group == "real_trained":
            labels[group] = "Real data"
        elif group.startswith("real_synthetic::"):
            labels[group] = "Real + " + group.split("::", 1)[1]
        elif group.startswith("synthetic::"):
            labels[group] = group.split("::", 1)[1]
        else:
            labels[group] = group
    return labels


def _learning_logs(
    artifacts: SuiteArtifacts,
    *,
    window_id: str,
    ticker_group: str | None,
    group: str,
    commission_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_table = artifacts.read_window_csv(
        window_id, "run_table.csv", ticker_group=ticker_group
    )
    selected = run_table.loc[
        (run_table["group"] == group)
        & (run_table["commission_name"] == commission_name)
    ]
    if selected.empty:
        raise ValueError(f"No runs for {window_id}, {group}, {commission_name}")
    training_frames = []
    validation_frames = []
    for _, row in selected.iterrows():
        cache_dir = Path(str(row["cache_dir"])).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = artifacts.manifest_path.parent / cache_dir
        training_path = cache_dir / "training_log.csv"
        validation_path = cache_dir / "validation_log.csv"
        if not training_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(
                f"Learning-curve logs are missing from cache directory: {cache_dir}"
            )
        training = pd.read_csv(training_path)
        validation = pd.read_csv(validation_path)
        training["seed"] = int(row["seed"])
        validation["seed"] = int(row["seed"])
        training_frames.append(training)
        validation_frames.append(validation)
    return (
        pd.concat(training_frames, ignore_index=True),
        pd.concat(validation_frames, ignore_index=True),
    )


def plot_learning_curves(
    artifacts: SuiteArtifacts,
    *,
    window_id: str,
    ticker_group: str | None = None,
    group: str,
    commission_name: str = "with_fee",
    smoothing_episodes: int = 10,
):
    """Plot per-seed training rewards and mean validation diagnostics."""
    if smoothing_episodes < 1:
        raise ValueError("smoothing_episodes must be at least 1")
    training, validation = _learning_logs(
        artifacts,
        window_id=window_id,
        ticker_group=ticker_group,
        group=group,
        commission_name=commission_name,
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_axis, validation_axis = axes
    for seed, frame in training.groupby("seed", sort=True):
        frame = frame.sort_values("timesteps")
        smoothed = (
            frame["episode_reward"].rolling(smoothing_episodes, min_periods=1).mean()
        )
        train_axis.plot(
            frame["timesteps"],
            smoothed,
            linewidth=1.4,
            alpha=0.8,
            label=f"seed {seed}",
        )
    train_axis.set_title(f"Training episode reward ({smoothing_episodes}-episode mean)")
    train_axis.set_xlabel("Timesteps")
    train_axis.set_ylabel("Episode reward")
    train_axis.grid(alpha=0.2)
    train_axis.legend()

    validation_series = [
        ("train_reward_mean", "Train evaluation", "#0057B8"),
        ("real_valid_reward", "Real validation", "#E66100"),
    ]
    if validation["synthetic_valid_reward_mean"].notna().any():
        validation_series.append(
            ("synthetic_valid_reward_mean", "Synthetic validation", "#009E73")
        )
    for column, label, color in validation_series:
        summary = (
            validation.groupby("timesteps", sort=True)[column]
            .agg(["mean", "std"])
            .reset_index()
        )
        x_values = summary["timesteps"].to_numpy(dtype=float)
        means = summary["mean"].to_numpy(dtype=float)
        deviations = summary["std"].fillna(0).to_numpy(dtype=float)
        validation_axis.plot(x_values, means, color=color, linewidth=2, label=label)
        validation_axis.fill_between(
            x_values,
            means - deviations,
            means + deviations,
            color=color,
            alpha=0.14,
        )
    validation_axis.set_title("Evaluation reward (mean ± seed SD)")
    validation_axis.set_xlabel("Timesteps")
    validation_axis.set_ylabel("Reward per day")
    validation_axis.grid(alpha=0.2)
    validation_axis.legend()
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    fig.suptitle(f"{window_id} | {selected_ticker_group}: {group} | {commission_name}")
    fig.tight_layout()
    return fig, axes


def plot_test_metric_by_window(
    artifacts: SuiteArtifacts,
    *,
    metric: str = "sharpe",
    commission_name: str = "with_fee",
    ticker_group: str | None = None,
    group_labels: Mapping[str, str] | None = None,
):
    """Plot seed means and standard deviations for every test window."""
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    required = {
        "window_id",
        "ticker_group",
        "commission_name",
        "group",
        mean_column,
        std_column,
    }
    missing = sorted(required - set(artifacts.aggregate_summary.columns))
    if missing:
        raise ValueError(f"aggregate_summary is missing {missing}")
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    frame = artifacts.aggregate_summary.loc[
        (artifacts.aggregate_summary["commission_name"] == commission_name)
        & (artifacts.aggregate_summary["ticker_group"] == selected_ticker_group)
    ].copy()
    if frame.empty:
        raise ValueError(f"No results for commission {commission_name!r}")
    groups = list(dict.fromkeys(frame["group"]))
    windows = list(dict.fromkeys(frame["window_id"]))
    labels = dict(group_labels or _default_group_labels(groups))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    width = 0.8 / max(len(groups), 1)
    x_positions = np.arange(len(windows), dtype=float)
    fig, axis = plt.subplots(figsize=(max(10, 2.4 * len(windows)), 6))
    for index, (group, color) in enumerate(zip(groups, colors)):
        group_rows = frame.loc[frame["group"] == group].set_index("window_id")
        values = group_rows.reindex(windows)[mean_column].to_numpy(dtype=float)
        errors = group_rows.reindex(windows)[std_column].fillna(0).to_numpy(dtype=float)
        offset = (index - (len(groups) - 1) / 2) * width
        axis.errorbar(
            x_positions + offset,
            values,
            yerr=errors,
            fmt="o",
            capsize=3,
            color=color,
            label=labels.get(group, group),
        )
    axis.axhline(0, color="#6B7280", linewidth=0.8)
    axis.set_xticks(x_positions, windows, rotation=20, ha="right")
    axis.set_ylabel(f"Test {metric} (mean ± seed SD)")
    axis.set_title(
        f"Synthetic-vs-Real test {metric} | {selected_ticker_group} | "
        f"{commission_name}"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis


def plot_paired_metric_deltas(
    artifacts: SuiteArtifacts,
    *,
    metric: str = "sharpe",
    commission_name: str = "with_fee",
    ticker_group: str | None = None,
):
    """Plot each seed's paired delta versus the matching real-only policy."""
    column = f"delta_{metric}"
    required = {
        "window_id",
        "ticker_group",
        "commission_name",
        "training_group_label",
        column,
    }
    missing = sorted(required - set(artifacts.paired_deltas.columns))
    if missing:
        raise ValueError(f"paired_deltas is missing {missing}")
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    frame = artifacts.paired_deltas.loc[
        (artifacts.paired_deltas["commission_name"] == commission_name)
        & (artifacts.paired_deltas["ticker_group"] == selected_ticker_group)
    ].copy()
    if frame.empty:
        raise ValueError(f"No paired deltas for commission {commission_name!r}")
    frame["comparison"] = (
        frame["window_id"].astype(str)
        + " | "
        + frame["training_group_label"].astype(str)
    )
    order = list(dict.fromkeys(frame["comparison"]))
    y_lookup = {label: index for index, label in enumerate(order)}
    fig, axis = plt.subplots(figsize=(11, max(5, 0.42 * len(order))))
    for _, row in frame.iterrows():
        axis.scatter(
            float(row[column]),
            y_lookup[row["comparison"]],
            alpha=0.65,
            color="#0057B8",
        )
    medians = frame.groupby("comparison", sort=False)[column].median()
    for label, value in medians.items():
        axis.scatter(
            float(value),
            y_lookup[label],
            marker="D",
            s=55,
            color="#E66100",
            edgecolor="#111827",
            zorder=3,
        )
    axis.axvline(0, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.set_yticks(range(len(order)), order)
    axis.invert_yaxis()
    axis.set_xlabel(f"Paired test Δ {metric} versus real-only")
    axis.set_title(
        f"Paired Synthetic-vs-Real effects | {selected_ticker_group} | "
        f"{commission_name}"
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, axis


def plot_continuous_equity_curves(
    artifacts: SuiteArtifacts,
    *,
    ticker_group: str | None = None,
    commission_name: str = "with_fee",
):
    """Plot one median-Sharpe continuous OOS chain per training group."""
    if artifacts.continuous_metrics.empty or artifacts.continuous_equity_curves.empty:
        raise ValueError("Suite has no continuous evaluation artifacts")
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    metrics = artifacts.continuous_metrics.loc[
        (artifacts.continuous_metrics["ticker_group"] == selected_ticker_group)
        & (artifacts.continuous_metrics["commission_name"] == commission_name)
    ].copy()
    curves = artifacts.continuous_equity_curves.loc[
        (artifacts.continuous_equity_curves["ticker_group"] == selected_ticker_group)
        & (artifacts.continuous_equity_curves["commission_name"] == commission_name)
    ].copy()
    if metrics.empty or curves.empty:
        raise ValueError(
            f"No continuous results for {selected_ticker_group}, {commission_name}"
        )
    representatives = []
    for group, rows in metrics.groupby("group", sort=False):
        candidates = rows.dropna(subset=["sharpe"]).copy()
        if candidates.empty:
            selected = rows.sort_values("seed").iloc[len(rows) // 2]
        else:
            median = candidates["sharpe"].median()
            selected = candidates.loc[(candidates["sharpe"] - median).abs().idxmin()]
        representatives.append((group, int(selected["seed"])))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(representatives), 1)))
    fig, axis = plt.subplots(figsize=(13, 6))
    for (group, seed), color in zip(representatives, colors):
        curve = curves.loc[
            (curves["group"] == group) & (curves["seed"] == seed)
        ].sort_values("date")
        if curve.empty:
            continue
        axis.plot(
            pd.to_datetime(curve["date"]),
            curve["growth_of_one"],
            color=color,
            linewidth=1.8,
            label=f"{curve['agent'].iloc[0]} (seed {seed})",
        )
    axis.axhline(1, color="#6B7280", linewidth=0.8)
    axis.set_xlabel("Real out-of-sample date")
    axis.set_ylabel("Growth of $1")
    axis.set_title(
        f"Continuous walk-forward equity | {selected_ticker_group} | "
        f"{commission_name}"
    )
    axis.grid(alpha=0.2)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis


def plot_representative_equity_curves(
    artifacts: SuiteArtifacts,
    *,
    window_id: str,
    ticker_group: str | None = None,
    commission_name: str = "with_fee",
    period: str = "test",
):
    """Plot the median-test-Sharpe representative policy for each group."""
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    equity = artifacts.read_window_csv(
        window_id, "equity_curves.csv", ticker_group=selected_ticker_group
    )
    representatives = artifacts.read_window_csv(
        window_id,
        "median_sharpe_representatives.csv",
        ticker_group=selected_ticker_group,
    )
    subset = equity.loc[
        (equity["period"] == period) & (equity["commission_name"] == commission_name)
    ].copy()
    selected = representatives.loc[
        representatives["commission_name"] == commission_name
    ]
    if subset.empty or selected.empty:
        raise ValueError(f"No {period} equity data for {window_id}, {commission_name}")
    fig, axis = plt.subplots(figsize=(13, 6))
    groups = [group for group in selected["group"] if group != "buy_and_hold"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for group, color in zip(groups, colors):
        representative = selected.loc[selected["group"] == group].iloc[0]
        seed = int(representative["seed"])
        curve = subset.loc[
            (subset["group"] == group)
            & (pd.to_numeric(subset["seed"], errors="coerce") == seed)
        ].sort_values("date")
        if curve.empty:
            continue
        axis.plot(
            pd.to_datetime(curve["date"]),
            curve["growth_of_one"],
            linewidth=1.8,
            color=color,
            label=f"{curve['agent'].iloc[0]} (seed {seed})",
        )
    benchmark = subset.loc[subset["group"] == "buy_and_hold"].sort_values("date")
    if not benchmark.empty:
        axis.plot(
            pd.to_datetime(benchmark["date"]),
            benchmark["growth_of_one"],
            color="#111111",
            linestyle="--",
            linewidth=2,
            label="Buy & Hold",
        )
    axis.axhline(1, color="#6B7280", linewidth=0.8)
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel(f"Real {period} date")
    axis.set_title(
        f"{window_id} | {selected_ticker_group}: representative {period} "
        f"equity curves | {commission_name}"
    )
    axis.grid(alpha=0.2)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis


def plot_representative_portfolio_weights(
    artifacts: SuiteArtifacts,
    *,
    window_id: str,
    ticker_group: str | None = None,
    group: str,
    commission_name: str = "with_fee",
):
    """Plot actual end-of-day weights for one representative test policy."""
    selected_ticker_group = artifacts.select_ticker_group(ticker_group)
    weights = artifacts.read_window_csv(
        window_id, "portfolio_weights.csv", ticker_group=selected_ticker_group
    )
    representatives = artifacts.read_window_csv(
        window_id,
        "median_sharpe_representatives.csv",
        ticker_group=selected_ticker_group,
    )
    selected = representatives.loc[
        (representatives["group"] == group)
        & (representatives["commission_name"] == commission_name)
    ]
    if selected.empty:
        raise ValueError(
            f"No representative for {window_id}, {group}, {commission_name}"
        )
    seed = int(selected.iloc[0]["seed"])
    frame = weights.loc[
        (weights["group"] == group)
        & (weights["commission_name"] == commission_name)
        & (weights["seed"] == seed)
    ].copy()
    if frame.empty:
        raise ValueError(
            f"No portfolio weights for {window_id}, {group}, {commission_name}, seed {seed}"
        )
    pivot = frame.pivot(index="date", columns="asset", values="actual_weight")
    pivot.index = pd.to_datetime(pivot.index)
    preferred_order = ["Cash"] if "Cash" in pivot.columns else []
    preferred_order.extend(sorted(asset for asset in pivot.columns if asset != "Cash"))
    pivot = pivot[preferred_order]
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.stackplot(pivot.index, pivot.to_numpy().T, labels=pivot.columns, alpha=0.85)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Actual portfolio weight")
    axis.set_xlabel("Real test date")
    axis.set_title(
        f"{window_id} | {selected_ticker_group}: {group} | "
        f"{commission_name} | seed {seed}"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis
