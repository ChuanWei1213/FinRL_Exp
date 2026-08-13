from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from finrl.experiments.synthetic_vs_real import ExperimentConfig
from finrl.experiments.synthetic_vs_real import ExperimentPaths
from finrl.experiments.synthetic_vs_real import PreparedWindow
from finrl.experiments.synthetic_vs_real import TimeSplit
from finrl.experiments.synthetic_vs_real import WindowResult
from finrl.experiments.synthetic_vs_real import discover_synthetic_datasets
from finrl.experiments.synthetic_vs_real import environment_kwargs
from finrl.experiments.synthetic_vs_real import evaluate_continuous_test_chains
from finrl.experiments.synthetic_vs_real import prepare_window_data
from finrl.experiments.synthetic_vs_real import rolling_time_splits
from finrl.experiments.synthetic_vs_real import rollout
from finrl.experiments.synthetic_vs_real import training_cache_key
from training_cache_fingerprint import fingerprint_training_sources


TICKERS = ("AAA", "BBB")
TICKER_GROUP = "toy_2"
DATASET_ID = "rolling_toy_v1"


def time_split(**overrides) -> TimeSplit:
    values = {
        "name": "tiny_window",
        "train_start": "2024-01-01",
        "train_end": "2024-01-15",
        "val_start": "2024-01-15",
        "val_end": "2024-01-22",
        "test_start": "2024-01-22",
        "test_end": "2024-02-06",
    }
    values.update(overrides)
    return TimeSplit(**values)


def experiment_config(split: TimeSplit) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "real_data_glob": "data/real/*.csv",
            "synthetic_root": "data/synthetic",
            "artifact_dir": "results/test",
            "ticker_groups": {TICKER_GROUP: list(TICKERS)},
            "active_ticker_groups": [TICKER_GROUP],
            "synthetic_datasets": {TICKER_GROUP: DATASET_ID},
            "time_splits": [
                {
                    "name": split.name,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "val_start": split.val_start,
                    "val_end": split.val_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                }
            ],
            "synthetic_model_labels": {"toy__two_paths": "Toy"},
            "time_window": 3,
            "run_modes": {
                "smoke": {
                    "seeds": [0, 1],
                    "total_timesteps": 8,
                    "eval_freq": 4,
                    "commissions": {"with_fee": 0.0025},
                }
            },
        }
    )


def market_rows(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(dates):
        for ticker_index, ticker in enumerate(TICKERS):
            close = 10.0 + ticker_index * 5 + day * 0.1
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000 + day,
                }
            )
    return pd.DataFrame(rows)


def write_real_files(root: Path, frame: pd.DataFrame) -> list[Path]:
    real_dir = root / "data" / "real"
    real_dir.mkdir(parents=True)
    paths = []
    for ticker in TICKERS:
        path = real_dir / f"{ticker}.csv"
        subset = frame.loc[frame["tic"] == ticker].rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        subset[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
            path, index=False
        )
        paths.append(path)
    return paths


def write_synthetic_paths(
    root: Path,
    frame: pd.DataFrame,
    *,
    ticker_group: str = TICKER_GROUP,
    dataset_id: str = DATASET_ID,
) -> list[Path]:
    model_dir = (
        root / "data" / "synthetic" / ticker_group / dataset_id / "toy__two_paths"
    )
    model_dir.mkdir(parents=True)
    selected = frame[["date", "tic", "close", "high", "low"]]
    paths = []
    for index, multiplier in enumerate((1.0, 1.02)):
        path = model_dir / f"path_{index:03d}.csv"
        path_frame = selected.copy()
        path_frame[["close", "high", "low"]] *= multiplier
        path_frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_config_resolves_smoke_protocol_and_dataset():
    split = time_split()
    config = experiment_config(split)
    settings = config.resolve_run_settings("smoke")

    assert config.dataset_id_for(TICKER_GROUP) == DATASET_ID
    assert settings.seeds == (0, 1)
    assert settings.total_timesteps == 8
    assert settings.commission_map == {"with_fee": 0.0025}
    assert config.select_ticker_groups(None) == ((TICKER_GROUP, TICKERS),)
    assert config.evaluation_modes == ("independent", "continuous")


def test_config_selects_named_ticker_groups():
    split = time_split()
    config = ExperimentConfig.from_mapping(
        {
            "ticker_groups": {
                "tech_2": ["BBB", "AAA"],
                "chosen_3": ["DDD", "CCC", "AAA"],
            },
            "active_ticker_groups": ["tech_2"],
            "synthetic_datasets": {
                "tech_2": "tech_release",
                "chosen_3": "chosen_release",
            },
            "time_splits": [
                {
                    "name": split.name,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "val_start": split.val_start,
                    "val_end": split.val_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                }
            ],
        }
    )

    assert config.select_ticker_groups(None) == (("tech_2", ("AAA", "BBB")),)
    assert config.select_ticker_groups(["chosen_3", "tech_2"]) == (
        ("chosen_3", ("AAA", "CCC", "DDD")),
        ("tech_2", ("AAA", "BBB")),
    )


def test_discovery_and_preparation_use_dataset_directory(tmp_path):
    split = time_split()
    config = experiment_config(split)
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    synthetic_paths = write_synthetic_paths(tmp_path, frame)

    discovered = discover_synthetic_datasets(tmp_path, config, TICKER_GROUP)
    prepared = prepare_window_data(
        project_root=tmp_path,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )

    assert discovered == {"toy__two_paths": synthetic_paths}
    assert prepared.model_names == ("toy__two_paths",)
    assert prepared.ticker_group == TICKER_GROUP
    assert prepared.dataset_id == DATASET_ID
    assert prepared.tickers == TICKERS
    assert len(prepared.synthetic_pipelines["toy__two_paths"]["train_paths"]) == 2
    assert set(prepared.training_data_fingerprints) == {
        "real_trained",
        "synthetic::toy__two_paths",
        "real_synthetic::toy__two_paths",
    }


def test_rolling_schedule_uses_dataset_trading_days(tmp_path):
    config = ExperimentConfig.from_mapping(
        {
            "ticker_groups": {TICKER_GROUP: list(TICKERS)},
            "active_ticker_groups": [TICKER_GROUP],
            "synthetic_datasets": {TICKER_GROUP: DATASET_ID},
            "synthetic_root": "data/synthetic",
            "rolling_schedule": {
                "name": "toy_wf",
                "train_days": 6,
                "validation_days": 3,
                "test_days": 4,
                "step_days": 4,
                "include_partial_final_test": True,
            },
            "synthetic_model_labels": {"toy__two_paths": "Toy"},
        }
    )
    dates = pd.bdate_range("2024-01-01", periods=15)
    frame = market_rows(dates)
    write_synthetic_paths(tmp_path, frame)

    splits = rolling_time_splits(
        project_root=tmp_path,
        config=config,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )

    assert len(splits) == 2
    assert splits[0].train_start == dates[0].date().isoformat()
    assert splits[0].train_end == dates[6].date().isoformat()
    assert splits[0].val_end == dates[9].date().isoformat()
    assert splits[0].test_end == dates[13].date().isoformat()
    assert splits[1].test_start == dates[13].date().isoformat()
    assert splits[1].test_end == (dates[-1] + pd.Timedelta(days=1)).date().isoformat()


def test_dataset_layout_changes_training_fingerprint_identity(tmp_path):
    rows = pd.DataFrame(
        [
            {"date": "2024-01-02", "tic": "AAA", "close": 10, "high": 11, "low": 9},
            {"date": "2024-01-03", "tic": "AAA", "close": 11, "high": 12, "low": 10},
        ]
    )
    first = tmp_path / "data" / "synthetic" / "a" / "release_1" / "toy" / "path_001.csv"
    second = (
        tmp_path / "data" / "synthetic" / "a" / "release_2" / "toy" / "path_001.csv"
    )
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    rows.to_csv(first, index=False)
    rows.to_csv(second, index=False)

    first_hash = fingerprint_training_sources(
        project_root=tmp_path,
        sources=[("synthetic", first)],
        start="2024-01-01",
        end="2024-02-01",
    )
    second_hash = fingerprint_training_sources(
        project_root=tmp_path,
        sources=[("synthetic", second)],
        start="2024-01-01",
        end="2024-02-01",
    )

    assert first_hash != second_hash


def test_training_cache_key_ignores_test_dates_and_evaluation_mode(tmp_path):
    original_split = time_split()
    changed_test = original_split.with_test(
        test_start="2024-02-06", test_end="2024-02-20"
    )
    config = experiment_config(original_split)
    settings = config.resolve_run_settings("smoke")

    def prepared(split: TimeSplit) -> PreparedWindow:
        return PreparedWindow(
            split=split,
            ticker_group=TICKER_GROUP,
            dataset_id=DATASET_ID,
            tickers=TICKERS,
            real_files=[],
            synthetic_files_by_model={},
            real_pipeline={},
            synthetic_pipelines={},
            training_data_fingerprints={"real_trained": "same-data"},
            data_fingerprint="same-data",
            data_summary=pd.DataFrame(),
            scale_summary=pd.DataFrame(),
        )

    paths = ExperimentPaths(
        artifact_root=tmp_path,
        output_root=tmp_path,
        experiment_root=tmp_path / "experiment",
        train_cache_root=tmp_path / "cache",
        experiment_id="test",
    )
    first_key, first_payload = training_cache_key(
        config=config,
        settings=settings,
        prepared=prepared(original_split),
        paths=paths,
        group="real_trained",
        commission=0.0025,
        seed=0,
    )
    second_key, second_payload = training_cache_key(
        config=config,
        settings=settings,
        prepared=prepared(changed_test),
        paths=paths,
        group="real_trained",
        commission=0.0025,
        seed=0,
    )

    assert first_key == second_key
    assert first_payload == second_payload
    assert "test" not in first_payload
    assert "evaluation_modes" not in first_payload


def test_rollout_can_start_from_carried_actual_weights(tmp_path):
    config = experiment_config(time_split())
    frame = market_rows(pd.bdate_range("2024-01-01", periods=8))
    kwargs = environment_kwargs(config, 0.0025, cwd=tmp_path)
    initial_weights = np.array([0.2, 0.5, 0.3])

    result = rollout(
        frame[["date", "tic", "close", "high", "low"]],
        None,
        kwargs,
        seed=0,
        initial_portfolio_value=2_000,
        initial_actual_weights=initial_weights,
        initial_last_action=initial_weights,
    )

    assert result["values"][0] == 2_000
    np.testing.assert_allclose(result["actual_weights"][0], initial_weights)
    assert result["turnover_by_step"][0] > 0
    assert result["values"][-1] == result["final_portfolio_value"]


def test_continuous_evaluation_carries_previous_actual_weights(tmp_path, monkeypatch):
    first_split = time_split(
        name="first",
        test_start="2024-01-22",
        test_end="2024-02-06",
    )
    second_split = time_split(
        name="second",
        train_start="2024-01-08",
        train_end="2024-01-22",
        val_start="2024-01-22",
        val_end="2024-02-06",
        test_start="2024-02-06",
        test_end="2024-02-20",
    )
    config = experiment_config(first_split)
    settings = config.resolve_run_settings("smoke")
    observed_initial_weights = []

    def fake_rollout(
        frame,
        model,
        kwargs,
        seed,
        *,
        initial_portfolio_value=None,
        initial_actual_weights=None,
        initial_last_action=None,
    ):
        observed_initial_weights.append(np.asarray(initial_actual_weights).copy())
        start = float(initial_portfolio_value)
        final_actual = np.array([0.1, 0.6, 0.3])
        final_target = np.array([0.2, 0.5, 0.3])
        date = pd.Timestamp(frame["date"].iloc[-1])
        values = np.array([start, start * 1.01])
        return {
            "values": values,
            "dates": pd.DatetimeIndex([date - pd.Timedelta(days=1), date]),
            "target_weights": np.vstack([initial_last_action, final_target]),
            "actual_weights": np.vstack([initial_actual_weights, final_actual]),
            "turnover_by_step": np.array([0.4]),
            "trf_mu_by_step": np.array([0.999]),
            "weight_labels": ["Cash", *TICKERS],
            "metrics": {
                "final_value": values[-1],
                "cumulative_return": 0.01,
                "cagr": 0.01,
                "sharpe": 1.0,
                "annualized_volatility": 0.1,
                "max_drawdown": 0.0,
                "turnover": 0.4,
                "reward_per_day": 1.0,
            },
            "final_portfolio_value": values[-1],
            "final_actual_weights": final_actual,
            "final_target_weights": final_target,
        }

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.PPO.load", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr("finrl.experiments.synthetic_vs_real.rollout", fake_rollout)

    def window_result(split: TimeSplit) -> WindowResult:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp(split.test_end) - pd.Timedelta(days=1)],
                "tic": [TICKERS[0]],
            }
        )
        prepared = PreparedWindow(
            split=split,
            ticker_group=TICKER_GROUP,
            dataset_id=DATASET_ID,
            tickers=TICKERS,
            real_files=[],
            synthetic_files_by_model={},
            real_pipeline={"real_test": frame},
            synthetic_pipelines={},
            training_data_fingerprints={"real_trained": "same"},
            data_fingerprint="same",
            data_summary=pd.DataFrame(),
            scale_summary=pd.DataFrame(),
        )
        record = {
            "group": "real_trained",
            "model_name": None,
            "commission_name": "with_fee",
            "commission": 0.0025,
            "seed": 0,
            "best_model_path": tmp_path / "model",
            "best_timestep": 4,
        }
        return WindowResult(
            split=split,
            ticker_group=TICKER_GROUP,
            dataset_id=DATASET_ID,
            tickers=TICKERS,
            experiment_root=tmp_path,
            experiment_id=split.name,
            run_table=pd.DataFrame(),
            test_metrics=pd.DataFrame(),
            aggregate_summary=pd.DataFrame(),
            paired_deltas=pd.DataFrame(),
            representative_runs=pd.DataFrame(),
            prepared=prepared,
            run_records=[record],
        )

    artifacts = evaluate_continuous_test_chains(
        window_results=[window_result(first_split), window_result(second_split)],
        config=config,
        settings=settings,
    )

    np.testing.assert_allclose(observed_initial_weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(observed_initial_weights[1], [0.1, 0.6, 0.3])
    transitions = artifacts["window_transition_log"]
    assert transitions.iloc[1]["initial_portfolio_value"] == 101_000
    assert artifacts["continuous_metrics"].iloc[0]["test_days"] == 2
