from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from finrl.experiments.run_synthetic_vs_real import build_parser
from finrl.experiments.synthetic_vs_real import _independent_evaluation_item
from finrl.experiments.synthetic_vs_real import _concat_preserving_all_na_columns
from finrl.experiments.synthetic_vs_real import _content_hash
from finrl.experiments.synthetic_vs_real import _evaluation_environment_payload
from finrl.experiments.synthetic_vs_real import _legacy_independent_evaluation_payload
from finrl.experiments.synthetic_vs_real import _read_synthetic_path
from finrl.experiments.synthetic_vs_real import _require_same_dates
from finrl.experiments.synthetic_vs_real import _write_evaluation_cache
from finrl.experiments.synthetic_vs_real import INDEPENDENT_EVALUATION_ARTIFACTS
from finrl.experiments.synthetic_vs_real import CONTINUOUS_EVALUATION_ARTIFACTS
from finrl.experiments.synthetic_vs_real import (
    LEGACY_CONTINUOUS_EVALUATION_CACHE_SCHEMA,
)
from finrl.experiments.synthetic_vs_real import (
    LEGACY_INDEPENDENT_EVALUATION_CACHE_SCHEMA,
)
from finrl.experiments.synthetic_vs_real import ExperimentConfig
from finrl.experiments.synthetic_vs_real import ExperimentPaths
from finrl.experiments.synthetic_vs_real import PreparedWindow
from finrl.experiments.synthetic_vs_real import TimeSplit
from finrl.experiments.synthetic_vs_real import WindowResult
from finrl.experiments.synthetic_vs_real import discover_synthetic_datasets
from finrl.experiments.synthetic_vs_real import environment_kwargs
from finrl.experiments.synthetic_vs_real import evaluate_continuous_test_chains
from finrl.experiments.synthetic_vs_real import load_experiment_config
from finrl.experiments.synthetic_vs_real import legacy_training_cache_key
from finrl.experiments.synthetic_vs_real import plan_training_run
from finrl.experiments.synthetic_vs_real import prepare_window_data
from finrl.experiments.synthetic_vs_real import resolve_experiment_paths
from finrl.experiments.synthetic_vs_real import rolling_time_splits
from finrl.experiments.synthetic_vs_real import rollout
from finrl.experiments.synthetic_vs_real import run_experiment_suite
from finrl.experiments.synthetic_vs_real import run_training_matrix
from finrl.experiments.synthetic_vs_real import TrainingPlan
from finrl.experiments.synthetic_vs_real import TrainingCacheIndex
from finrl.experiments.synthetic_vs_real import training_cache_key
from finrl.experiments.synthetic_vs_real import training_groups
from training_cache_fingerprint import fingerprint_training_sources
from training_cache_fingerprint import fingerprint_semantic_training_sources
from finrl.experiments.notebook_utils import PriceScaleByTicker


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
            "time_window": 3,
            "studies": {
                "standard": {
                    "training_groups": [
                        "real_trained",
                        "synthetic",
                        "real_synthetic",
                    ],
                    "synthetic_models": {"toy__two_paths": "Toy"},
                    "run_modes": {
                        "smoke": {
                            "seeds": [0, 1],
                            "total_timesteps": 8,
                            "eval_freq": 4,
                            "commissions": {"with_fee": 0.0025},
                        }
                    },
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


def test_concat_preserves_all_na_columns_without_future_warning():
    ppo = pd.DataFrame(
        {
            "kind": ["ppo"],
            "synthetic_path_count": pd.Series([10], dtype="Int64"),
            "all_na": pd.Series([pd.NA], dtype="Int64"),
        }
    )
    benchmark = pd.DataFrame(
        {
            "kind": ["benchmark"],
            "synthetic_path_count": [None],
            "all_na": [None],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        combined = _concat_preserving_all_na_columns([ppo, benchmark])

    assert combined.columns.tolist() == [
        "kind",
        "synthetic_path_count",
        "all_na",
    ]
    assert combined["kind"].tolist() == ["ppo", "benchmark"]
    assert str(combined["synthetic_path_count"].dtype) == "Int64"
    assert combined["all_na"].isna().all()


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
    model_name: str = "toy__two_paths",
    count: int = 2,
) -> list[Path]:
    model_dir = root / "data" / "synthetic" / ticker_group / dataset_id / model_name
    model_dir.mkdir(parents=True)
    selected = frame[["date", "tic", "close", "high", "low"]]
    paths = []
    for index in range(count):
        multiplier = 1.0 + index * 0.02
        path = model_dir / f"path_{index:03d}.csv"
        path_frame = selected.copy()
        path_frame[["close", "high", "low"]] *= multiplier
        path_frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def path_count_experiment_config(
    split: TimeSplit,
    counts: tuple[int, ...] = (2, 5),
    *,
    equivalents: dict[int, str] | None = None,
) -> ExperimentConfig:
    raw = experiment_config(split).to_mapping()
    raw["studies"]["path-count"] = {
        "training_groups": ["real_trained", "real_synthetic"],
        "synthetic_path_subsets": {
            "toy__many_paths": {
                "label": "Toy",
                "counts": list(counts),
                **(
                    {
                        "equivalent_source_models": {
                            str(count): model for count, model in equivalents.items()
                        }
                    }
                    if equivalents
                    else {}
                ),
            }
        },
        "run_modes": {
            "smoke": {
                "seeds": [0],
                "total_timesteps": 4096,
                "eval_freq": 2048,
                "commissions": {"with_fee": 0.0025},
            },
            "full": {
                "seeds": [0],
                "total_timesteps": 200_000,
                "eval_freq": 20_000,
                "commissions": {"with_fee": 0.0025},
            },
        },
    }
    return ExperimentConfig.from_mapping(raw)


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


def test_config_requires_unified_study_schema():
    raw = experiment_config(time_split()).to_mapping()
    raw.pop("studies")
    with pytest.raises(ValueError, match="studies must contain"):
        ExperimentConfig.from_mapping(raw)

    raw = experiment_config(time_split()).to_mapping()
    raw["run_modes"] = {}
    with pytest.raises(ValueError, match="no longer supported"):
        ExperimentConfig.from_mapping(raw)


def test_path_count_study_resolves_variants_groups_and_budgets():
    config = path_count_experiment_config(time_split()).for_study("path-count")

    assert config.study_name == "path-count"
    assert config.training_group_kinds == ("real_trained", "real_synthetic")
    assert list(config.synthetic_variants) == [
        "toy__many_paths__first_2_paths",
        "toy__many_paths__first_5_paths",
    ]
    assert training_groups(config.synthetic_variants, config.training_group_kinds) == [
        "real_trained",
        "real_synthetic::toy__many_paths__first_2_paths",
        "real_synthetic::toy__many_paths__first_5_paths",
    ]

    smoke = config.resolve_run_settings("smoke")
    full = config.resolve_run_settings("full")
    assert smoke.seeds == full.seeds == (0,)
    assert smoke.commission_map == full.commission_map == {"with_fee": 0.0025}
    assert (smoke.total_timesteps, smoke.eval_freq) == (4_096, 2_048)
    assert (full.total_timesteps, full.eval_freq) == (200_000, 20_000)


def test_dataset_selection_supports_variant_or_source_model():
    standard = experiment_config(time_split()).for_study("standard")
    assert list(
        standard.select_synthetic_variants(
            include=["toy__two_paths"]
        ).synthetic_variants
    ) == ["toy__two_paths"]
    assert (
        standard.select_synthetic_variants(
            exclude=["toy__two_paths"]
        ).synthetic_variants
        == {}
    )

    path_count = path_count_experiment_config(time_split()).for_study("path-count")
    assert list(
        path_count.select_synthetic_variants(
            include=["toy__many_paths"]
        ).synthetic_variants
    ) == [
        "toy__many_paths__first_2_paths",
        "toy__many_paths__first_5_paths",
    ]
    assert list(
        path_count.select_synthetic_variants(
            include=["toy__many_paths__first_2_paths"]
        ).synthetic_variants
    ) == ["toy__many_paths__first_2_paths"]
    assert list(
        path_count.select_synthetic_variants(
            exclude=["toy__many_paths__first_5_paths"]
        ).synthetic_variants
    ) == ["toy__many_paths__first_2_paths"]


def test_dataset_selection_rejects_mixed_or_unknown_selectors():
    config = experiment_config(time_split()).for_study("standard")
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.select_synthetic_variants(
            include=["toy__two_paths"], exclude=["toy__two_paths"]
        )
    with pytest.raises(ValueError, match="Unknown dataset selector"):
        config.select_synthetic_variants(include=["missing"])


@pytest.mark.parametrize("counts", [(0, 2), (2, 2), (5, 2), (1.5, 2)])
def test_path_count_study_rejects_nonpositive_duplicate_or_unsorted_counts(counts):
    with pytest.raises(ValueError, match="counts must"):
        path_count_experiment_config(time_split(), counts)


def test_project_path_count_study_plans_all_configured_variants_per_window():
    project_root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        project_root / "configs" / "synthetic_vs_real.json"
    ).for_study("path-count")

    groups = training_groups(config.synthetic_variants, config.training_group_kinds)
    assert groups == [
        "real_trained",
        *[f"real_synthetic::{variant}" for variant in config.synthetic_variants],
    ]
    assert len(groups) == 1 + len(config.synthetic_variants)


def test_cli_defaults_to_all_stage_and_one_worker():
    args = build_parser().parse_args([])

    assert args.study == "standard"
    assert args.stage == "all"
    assert args.workers == 1
    assert build_parser().parse_args(["--study", "path-count"]).study == "path-count"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--workers", "0"])


def test_cli_dataset_include_and_exclude_are_mutually_exclusive():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.include_datasets is None
    assert defaults.exclude_datasets is None
    assert parser.parse_args(
        ["--include-dataset", "first", "--include-dataset", "second"]
    ).include_datasets == ["first", "second"]
    assert parser.parse_args(["--exclude-dataset", "first"]).exclude_datasets == [
        "first"
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["--include-dataset", "first", "--exclude-dataset", "second"])


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
            "studies": {
                "standard": {
                    "training_groups": [
                        "real_trained",
                        "synthetic",
                        "real_synthetic",
                    ],
                    "synthetic_models": {"toy": "Toy"},
                    "run_modes": {
                        "smoke": {
                            "seeds": [0],
                            "total_timesteps": 8,
                            "eval_freq": 4,
                            "commissions": {"with_fee": 0.0025},
                        }
                    },
                }
            },
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


def test_path_count_discovery_prepares_nested_subsets_once_with_local_scalers(
    tmp_path, monkeypatch
):
    split = time_split()
    config = path_count_experiment_config(split).for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    source_paths = write_synthetic_paths(
        tmp_path,
        frame,
        model_name="toy__many_paths",
        count=5,
    )
    discovered = discover_synthetic_datasets(tmp_path, config, TICKER_GROUP)

    assert discovered == {
        "toy__many_paths__first_2_paths": source_paths[:2],
        "toy__many_paths__first_5_paths": source_paths[:5],
    }

    read_paths = []

    def tracked_reader(path, tickers, label):
        read_paths.append(path)
        return _read_synthetic_path(path, tickers, label)

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._read_synthetic_path", tracked_reader
    )
    prepared = prepare_window_data(
        project_root=tmp_path,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )

    assert read_paths == source_paths
    assert prepared.study == "path-count"
    assert prepared.synthetic_files_by_model == discovered
    assert [
        len(prepared.synthetic_pipelines[model_name]["train_paths"])
        for model_name in prepared.model_names
    ] == [2, 5]
    assert [
        len(prepared.synthetic_pipelines[model_name]["synthetic_valid_paths"])
        for model_name in prepared.model_names
    ] == [2, config.max_synthetic_validation_paths]
    first_scaler = prepared.synthetic_pipelines["toy__many_paths__first_2_paths"][
        "scaler"
    ]
    second_scaler = prepared.synthetic_pipelines["toy__many_paths__first_5_paths"][
        "scaler"
    ]
    assert first_scaler.scales != second_scaler.scales
    assert set(prepared.training_data_fingerprints) == {
        "real_trained",
        "real_synthetic::toy__many_paths__first_2_paths",
        "real_synthetic::toy__many_paths__first_5_paths",
    }
    assert len(set(prepared.training_data_fingerprints.values())) == 3
    synthetic_train = prepared.data_summary.loc[
        prepared.data_summary["split"].eq("synthetic train")
    ]
    assert synthetic_train["paths"].tolist() == [2, 5]
    assert synthetic_train["synthetic_path_count"].tolist() == [2, 5]


def test_price_scaler_streaming_fit_and_vectorized_transform_match_legacy():
    dates = pd.bdate_range("2024-01-01", periods=5)
    first = market_rows(dates)
    second = market_rows(dates)
    second[["close", "high", "low"]] *= 1.7
    frames = [first, second]
    columns = ("close", "high", "low")
    pooled = pd.concat(frames, ignore_index=True)
    expected_scales = {
        ticker: float(
            np.abs(
                pooled.loc[pooled["tic"].eq(ticker), list(columns)].to_numpy(
                    dtype=float
                )
            ).max()
        )
        for ticker in sorted(TICKERS)
    }
    scaler = PriceScaleByTicker(TICKERS, columns).fit(frames)

    assert scaler.scales == expected_scales
    expected = first.copy()
    for ticker, scale in expected_scales.items():
        mask = expected["tic"].eq(ticker)
        expected.loc[mask, list(columns)] = expected.loc[mask, list(columns)] / scale
    pd.testing.assert_frame_equal(scaler.transform(first), expected)


def test_materialization_checks_each_unique_path_grid_once_and_reports_progress(
    tmp_path, monkeypatch
):
    split = time_split()
    config = path_count_experiment_config(split).for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    write_synthetic_paths(
        tmp_path,
        frame,
        model_name="toy__many_paths",
        count=5,
    )
    checked_labels = []
    progress_descriptions = []

    def tracked_same_dates(expected, actual, *, label):
        checked_labels.append(label)
        return _require_same_dates(expected, actual, label=label)

    def tracked_tqdm(iterable=None, **kwargs):
        progress_descriptions.append(kwargs.get("desc", ""))
        return iterable

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._require_same_dates",
        tracked_same_dates,
    )
    monkeypatch.setattr("finrl.experiments.synthetic_vs_real.tqdm", tracked_tqdm)
    prepare_window_data(
        project_root=tmp_path,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
        show_progress=True,
    )

    synthetic_grid_checks = [label for label in checked_labels if "synthetic" in label]
    assert len(synthetic_grid_checks) == 10
    assert len(set(synthetic_grid_checks)) == 10
    assert any("reading and validating" in value for value in progress_descriptions)
    assert any("slicing and checking" in value for value in progress_descriptions)
    assert any("computing per-path scaler" in value for value in progress_descriptions)
    assert any(
        "transforming subset variants" in value for value in progress_descriptions
    )


def test_equivalent_source_model_is_verified_and_shares_training_identity(
    tmp_path, monkeypatch
):
    split = time_split()
    base_config = path_count_experiment_config(
        split, counts=(2,), equivalents={2: "toy__two_paths"}
    )
    standard = base_config.for_study("standard")
    path_count = base_config.for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    write_synthetic_paths(tmp_path, frame, model_name="toy__two_paths", count=2)
    write_synthetic_paths(tmp_path, frame, model_name="toy__many_paths", count=2)

    standard_prepared = prepare_window_data(
        project_root=tmp_path,
        config=standard,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )
    path_prepared = prepare_window_data(
        project_root=tmp_path,
        config=path_count,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )
    standard_group = "real_synthetic::toy__two_paths"
    path_group = "real_synthetic::toy__many_paths__first_2_paths"

    assert (
        standard_prepared.training_data_fingerprints[standard_group]
        == path_prepared.training_data_fingerprints[path_group]
    )
    alias = path_prepared.legacy_training_cache_aliases[path_group]
    assert alias.group == standard_group
    assert alias.equivalent_source_model == "toy__two_paths"

    settings = path_count.resolve_run_settings("smoke")
    standard_paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=standard,
        settings=settings,
        prepared=standard_prepared,
    )
    path_paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=path_count,
        settings=settings,
        prepared=path_prepared,
    )
    standard_key, _ = training_cache_key(
        config=standard,
        settings=settings,
        prepared=standard_prepared,
        paths=standard_paths,
        group=standard_group,
        commission=0.0025,
        seed=0,
    )
    path_key, _ = training_cache_key(
        config=path_count,
        settings=settings,
        prepared=path_prepared,
        paths=path_paths,
        group=path_group,
        commission=0.0025,
        seed=0,
    )
    assert standard_key == path_key

    legacy = legacy_training_cache_key(
        config=path_count,
        settings=settings,
        prepared=path_prepared,
        paths=path_paths,
        group=path_group,
        commission=0.0025,
        seed=0,
    )
    assert legacy is not None
    legacy_key, legacy_payload, _ = legacy
    old_cache_dir = path_paths.train_cache_root / "old-standard-cache"
    old_cache_dir.mkdir(parents=True)
    (old_cache_dir / "best_model.zip").write_bytes(b"best")
    (old_cache_dir / "last_model.zip").write_bytes(b"last")
    (old_cache_dir / "training_log.csv").write_text("reward\n1\n")
    (old_cache_dir / "validation_log.csv").write_text("reward\n1\n")
    (old_cache_dir / "status.json").write_text(
        json.dumps(
            {
                "completed": True,
                "cache_key": legacy_key,
                "semantic_cache_config": legacy_payload,
                "group": standard_group,
                "run_id": "old-standard-run",
                "best_timestep": 2048,
            }
        )
    )

    plan = plan_training_run(
        config=path_count,
        settings=settings,
        prepared=path_prepared,
        paths=path_paths,
        group=path_group,
        commission_name="with_fee",
        commission=0.0025,
        seed=0,
        use_cache=True,
        force_retrain=False,
    )
    assert plan.cache_match is not None
    assert plan.cache_match[0] == old_cache_dir / "status.json"
    assert plan.cache_match_type == "legacy_alias"
    assert plan.cache_equivalent_model == "toy__two_paths"
    alias_path = plan.planned_run_dir / "semantic_alias.json"
    assert alias_path.is_file()
    assert not (plan.planned_run_dir / "best_model.zip").exists()

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.resolve_legacy_training_cache_aliases",
        lambda **kwargs: pytest.fail("semantic alias recalculated legacy fingerprint"),
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.legacy_training_cache_key",
        lambda **kwargs: pytest.fail("semantic alias rebuilt legacy cache key"),
    )
    second_plan = plan_training_run(
        config=path_count,
        settings=settings,
        prepared=path_prepared,
        paths=path_paths,
        group=path_group,
        commission_name="with_fee",
        commission=0.0025,
        seed=0,
        use_cache=True,
        force_retrain=False,
        cache_index=TrainingCacheIndex(path_paths.train_cache_root),
    )
    assert second_plan.cache_match is not None
    assert second_plan.cache_match[0] == old_cache_dir / "status.json"
    assert second_plan.cache_match_type == "semantic_alias"


def test_equivalent_source_model_rejects_different_ordered_content(tmp_path):
    split = time_split()
    config = path_count_experiment_config(
        split, counts=(2,), equivalents={2: "toy__two_paths"}
    ).for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    write_synthetic_paths(tmp_path, frame, model_name="toy__two_paths", count=2)
    many_paths = write_synthetic_paths(
        tmp_path, frame, model_name="toy__many_paths", count=2
    )
    changed = pd.read_csv(many_paths[0])
    changed.loc[0, "close"] += 1
    changed.to_csv(many_paths[0], index=False)

    with pytest.raises(ValueError, match="ordered canonical paths differ"):
        prepare_window_data(
            project_root=tmp_path,
            config=config,
            split=split,
            real_files=real_files,
            real_raw=frame,
            ticker_group=TICKER_GROUP,
            tickers=TICKERS,
        )


def test_equivalent_source_model_requires_exact_path_count(tmp_path):
    config = path_count_experiment_config(
        time_split(), counts=(2,), equivalents={2: "toy__two_paths"}
    ).for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    write_synthetic_paths(tmp_path, frame, model_name="toy__two_paths", count=3)
    write_synthetic_paths(tmp_path, frame, model_name="toy__many_paths", count=2)

    with pytest.raises(ValueError, match="must contain exactly 2 paths"):
        discover_synthetic_datasets(tmp_path, config, TICKER_GROUP)


def test_path_count_discovery_fails_when_source_has_too_few_paths(tmp_path):
    config = path_count_experiment_config(time_split()).for_study("path-count")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    write_synthetic_paths(
        tmp_path,
        frame,
        model_name="toy__many_paths",
        count=4,
    )

    with pytest.raises(ValueError, match="requires the first 5 paths.*only 4"):
        discover_synthetic_datasets(tmp_path, config, TICKER_GROUP)


def test_path_count_artifacts_are_isolated_but_standard_paths_are_unchanged(tmp_path):
    split = time_split()
    standard = experiment_config(split)
    path_count = path_count_experiment_config(split).for_study("path-count")

    def prepared(config: ExperimentConfig) -> PreparedWindow:
        return PreparedWindow(
            split=split,
            ticker_group=TICKER_GROUP,
            dataset_id=DATASET_ID,
            tickers=TICKERS,
            real_files=[],
            synthetic_files_by_model={name: [] for name in config.synthetic_variants},
            real_pipeline={},
            synthetic_pipelines={},
            training_data_fingerprints={},
            data_fingerprint="same",
            data_summary=pd.DataFrame(),
            scale_summary=pd.DataFrame(),
            study=config.study_name,
        )

    standard_paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=standard,
        settings=standard.resolve_run_settings("smoke"),
        prepared=prepared(standard),
    )
    path_count_paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=path_count,
        settings=path_count.resolve_run_settings("smoke"),
        prepared=prepared(path_count),
    )

    artifact_root = tmp_path / "results" / "test"
    assert standard_paths.output_root.parent.parent == artifact_root / "smoke"
    assert path_count_paths.output_root.parent.parent == (
        artifact_root / "path-count" / "smoke"
    )
    assert standard_paths.train_cache_root == path_count_paths.train_cache_root


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
            "studies": {
                "standard": {
                    "training_groups": [
                        "real_trained",
                        "synthetic",
                        "real_synthetic",
                    ],
                    "synthetic_models": {"toy__two_paths": "Toy"},
                    "run_modes": {
                        "smoke": {
                            "seeds": [0],
                            "total_timesteps": 8,
                            "eval_freq": 4,
                            "commissions": {"with_fee": 0.0025},
                        }
                    },
                }
            },
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

    first_semantic = fingerprint_semantic_training_sources(
        sources=[("synthetic", first)],
        start="2024-01-01",
        end="2024-02-01",
    )
    second_semantic = fingerprint_semantic_training_sources(
        sources=[("synthetic", second)],
        start="2024-01-01",
        end="2024-02-01",
    )

    assert first_semantic == second_semantic


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


def test_independent_evaluation_cache_skips_completed_rollout(tmp_path, monkeypatch):
    split = time_split()
    config = experiment_config(split)
    settings = config.resolve_run_settings("smoke")
    model_path = tmp_path / "model"
    model_path.with_suffix(".zip").write_bytes(b"model-v1")
    evaluation_frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "tic": [TICKERS[0]],
            "close": [1.0],
            "high": [1.0],
            "low": [1.0],
        }
    )
    prepared = PreparedWindow(
        split=split,
        ticker_group=TICKER_GROUP,
        dataset_id=DATASET_ID,
        tickers=TICKERS,
        real_files=[],
        synthetic_files_by_model={},
        real_pipeline={
            "train": evaluation_frame,
            "real_valid": evaluation_frame,
            "real_test": evaluation_frame,
        },
        synthetic_pipelines={},
        training_data_fingerprints={"real_trained": "train-data"},
        data_fingerprint="evaluation-data",
        data_summary=pd.DataFrame(),
        scale_summary=pd.DataFrame(),
    )
    paths = ExperimentPaths(
        artifact_root=tmp_path,
        output_root=tmp_path,
        experiment_root=tmp_path / "experiment",
        train_cache_root=tmp_path / "train_cache",
        evaluation_cache_root=tmp_path / "evaluation_cache",
        experiment_id="test",
    )
    record = {
        "run_id": "real_trained__with_fee__seed_0",
        "group": "real_trained",
        "model_name": None,
        "commission_name": "with_fee",
        "commission": 0.0025,
        "seed": 0,
        "best_timestep": 4,
        "best_model_path": model_path,
        "cache_key": "training-key",
    }
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs["record"]["run_id"])
        return {
            "period_metrics": pd.DataFrame([{"period": "test", "value": 1.0}]),
            "equity_curves": pd.DataFrame(
                [{"date": pd.Timestamp("2024-01-01"), "growth_of_one": 1.0}]
            ),
            "portfolio_weights": pd.DataFrame(
                [{"date": pd.Timestamp("2024-01-01"), "asset": "Cash"}]
            ),
        }

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._evaluate_policy_independent",
        fake_evaluate,
    )
    first, first_timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=True,
        force_reevaluate=False,
    )
    second, second_timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=True,
        force_reevaluate=False,
    )

    assert calls == [record["run_id"]]
    assert first_timing["evaluation_cache_hit"] is False
    assert second_timing["evaluation_cache_hit"] is True
    pd.testing.assert_frame_equal(first["period_metrics"], second["period_metrics"])

    _, forced_timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=True,
        force_reevaluate=True,
    )

    assert calls == [record["run_id"], record["run_id"]]
    assert forced_timing["evaluation_cache_hit"] is False

    (Path(forced_timing["evaluation_cache_dir"]) / "equity_curves.csv").unlink()
    _, repaired_timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=True,
        force_reevaluate=False,
    )

    assert calls == [record["run_id"], record["run_id"], record["run_id"]]
    assert repaired_timing["evaluation_cache_hit"] is False

    uncached_root = tmp_path / "disabled_evaluation_cache"
    uncached_paths = ExperimentPaths(
        artifact_root=tmp_path,
        output_root=tmp_path,
        experiment_root=tmp_path / "uncached_experiment",
        train_cache_root=tmp_path / "disabled_train_cache",
        evaluation_cache_root=uncached_root,
        experiment_id="uncached",
    )
    _, uncached_timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=uncached_paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=False,
        force_reevaluate=False,
    )

    assert uncached_timing["evaluation_cache_dir"] == ""
    assert not uncached_root.exists()


def test_path_count_independent_evaluation_reuses_legacy_standard_cache(
    tmp_path, monkeypatch
):
    split = time_split()
    config = path_count_experiment_config(
        split, counts=(2,), equivalents={2: "toy__two_paths"}
    ).for_study("path-count")
    settings = config.resolve_run_settings("smoke")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    write_synthetic_paths(tmp_path, frame, model_name="toy__two_paths", count=2)
    write_synthetic_paths(tmp_path, frame, model_name="toy__many_paths", count=2)
    prepared = prepare_window_data(
        project_root=tmp_path,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )
    paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=config,
        settings=settings,
        prepared=prepared,
    )
    group = "real_synthetic::toy__many_paths__first_2_paths"
    legacy = legacy_training_cache_key(
        config=config,
        settings=settings,
        prepared=prepared,
        paths=paths,
        group=group,
        commission=0.0025,
        seed=0,
    )
    assert legacy is not None
    legacy_training_key = legacy[0]
    model_path = tmp_path / "legacy_model"
    model_bytes = b"legacy-model"
    model_path.with_suffix(".zip").write_bytes(model_bytes)
    record = {
        "run_id": "current-path-count-run",
        "group": group,
        "model_name": "toy__many_paths__first_2_paths",
        "commission_name": "with_fee",
        "commission": 0.0025,
        "seed": 0,
        "best_timestep": 2048,
        "best_model_path": model_path,
        "cache_key": legacy_training_key,
        "training_cache_source_group": "real_synthetic::toy__two_paths",
        "training_cache_source_run_id": "legacy-standard-run",
    }
    legacy_payload = _legacy_independent_evaluation_payload(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        model_hash=hashlib.sha256(model_bytes).hexdigest(),
    )
    assert legacy_payload is not None
    legacy_key = _content_hash(legacy_payload)
    cache_dir = (
        paths.evaluation_cache_root
        / "independent"
        / f"{split.name}__{TICKER_GROUP}"
        / f"legacy-standard-run__{legacy_key[:12]}"
    )
    legacy_frames = {
        "period_metrics": pd.DataFrame(
            [
                {
                    "run_id": "legacy-standard-run",
                    "period": "test",
                    "agent": "Real + Toy",
                    "group": "real_synthetic::toy__two_paths",
                    "model_name": "toy__two_paths",
                    "commission_name": "with_fee",
                    "commission": 0.0025,
                    "seed": 0,
                    "best_timestep": 2048,
                    "cumulative_return": 0.1,
                }
            ]
        ),
        "equity_curves": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-22"),
                    "group": "real_synthetic::toy__two_paths",
                    "model_name": "toy__two_paths",
                    "growth_of_one": 1.1,
                }
            ]
        ),
        "portfolio_weights": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-22"),
                    "group": "real_synthetic::toy__two_paths",
                    "model_name": "toy__two_paths",
                    "asset": "Cash",
                    "target_weight": 1.0,
                }
            ]
        ),
    }
    _write_evaluation_cache(
        cache_dir=cache_dir,
        cache_key=legacy_key,
        schema=LEGACY_INDEPENDENT_EVALUATION_CACHE_SCHEMA,
        artifact_names=INDEPENDENT_EVALUATION_ARTIFACTS,
        frames=legacy_frames,
        metadata={"semantic_cache_config": legacy_payload},
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._evaluate_policy_independent",
        lambda **kwargs: pytest.fail("legacy evaluation cache should be reused"),
    )

    frames, timing = _independent_evaluation_item(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        record=record,
        commission_name="with_fee",
        commission=0.0025,
        use_cache=True,
        force_reevaluate=False,
    )

    assert timing["evaluation_cache_hit"] is True
    assert timing["evaluation_cache_match_type"] == "legacy_alias"
    assert frames["period_metrics"].iloc[0]["run_id"] == record["run_id"]
    assert frames["period_metrics"].iloc[0]["group"] == group
    assert frames["period_metrics"].iloc[0]["model_name"] == record["model_name"]


def test_evaluate_stage_reports_all_missing_training_runs(tmp_path):
    split = time_split()
    config = experiment_config(split)
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    write_real_files(tmp_path, frame)
    write_synthetic_paths(tmp_path, frame)

    with pytest.raises(RuntimeError, match="Run --stage train first") as error:
        run_experiment_suite(
            project_root=tmp_path,
            config=config,
            mode="smoke",
            stage="evaluate",
            workers=4,
            show_progress=False,
        )

    message = str(error.value)
    assert "real_trained" in message
    assert "synthetic_toy__two_paths" in message


def test_path_count_evaluate_stage_requests_only_real_and_augmented_runs(tmp_path):
    split = time_split()
    config = path_count_experiment_config(split)
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    write_real_files(tmp_path, frame)
    write_synthetic_paths(
        tmp_path,
        frame,
        model_name="toy__many_paths",
        count=5,
    )

    with pytest.raises(RuntimeError, match="Run --stage train first") as error:
        run_experiment_suite(
            project_root=tmp_path,
            config=config,
            study="path-count",
            mode="smoke",
            stage="evaluate",
            workers=1,
            show_progress=False,
        )

    message = str(error.value)
    assert "real_trained" in message
    assert "real_synthetic_toy__many_paths__first_2_paths" in message
    assert "real_synthetic_toy__many_paths__first_5_paths" in message
    assert all("/synthetic_" not in run_id for run_id in error.value.run_ids)


def test_suite_dataset_filter_changes_training_matrix(tmp_path):
    split = time_split()
    config = path_count_experiment_config(split)
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    write_real_files(tmp_path, frame)
    write_synthetic_paths(
        tmp_path,
        frame,
        model_name="toy__many_paths",
        count=5,
    )

    with pytest.raises(RuntimeError, match="Run --stage train first") as included:
        run_experiment_suite(
            project_root=tmp_path,
            config=config,
            study="path-count",
            mode="smoke",
            stage="evaluate",
            workers=1,
            show_progress=False,
            include_datasets=["toy__many_paths__first_2_paths"],
        )
    included_message = str(included.value)
    assert "real_trained" in included_message
    assert "first_2_paths" in included_message
    assert "first_5_paths" not in included_message

    with pytest.raises(RuntimeError, match="Run --stage train first") as excluded:
        run_experiment_suite(
            project_root=tmp_path,
            config=config,
            study="path-count",
            mode="smoke",
            stage="evaluate",
            workers=1,
            show_progress=False,
            exclude_datasets=["toy__many_paths"],
        )
    assert len(excluded.value.run_ids) == 1
    assert "real_trained" in excluded.value.run_ids[0]


def test_parallel_training_submits_only_cache_misses_and_preserves_plan_order(
    tmp_path, monkeypatch
):
    split = time_split()
    config = experiment_config(split)
    settings = config.resolve_run_settings("smoke")
    prepared = PreparedWindow(
        split=split,
        ticker_group=TICKER_GROUP,
        dataset_id=DATASET_ID,
        tickers=TICKERS,
        real_files=[],
        synthetic_files_by_model={"toy__two_paths": []},
        real_pipeline={},
        synthetic_pipelines={},
        training_data_fingerprints={
            "real_trained": "real",
            "synthetic::toy__two_paths": "synthetic",
            "real_synthetic::toy__two_paths": "combined",
        },
        data_fingerprint="data",
        data_summary=pd.DataFrame(),
        scale_summary=pd.DataFrame(),
    )
    paths = ExperimentPaths(
        artifact_root=tmp_path,
        output_root=tmp_path,
        experiment_root=tmp_path / "experiment",
        train_cache_root=tmp_path / "train_cache",
        evaluation_cache_root=tmp_path / "evaluation_cache",
        experiment_id="parallel",
    )
    planned_ids = []

    def fake_plan(**kwargs):
        identifier = f"{kwargs['group']}::{kwargs['commission_name']}::{kwargs['seed']}"
        planned_ids.append(identifier)
        cache_match = (
            (tmp_path / "status.json", {"completed": True})
            if len(planned_ids) == 1
            else None
        )
        return TrainingPlan(
            group=kwargs["group"],
            commission_name=kwargs["commission_name"],
            commission=kwargs["commission"],
            seed=kwargs["seed"],
            identifier=identifier,
            cache_key=f"key-{len(planned_ids)}",
            cache_config={},
            planned_run_dir=tmp_path / identifier.replace(":", "_"),
            cache_match=cache_match,
            use_cache=True,
            force_retrain=False,
        )

    def record_for(plan, cache_hit):
        return {
            "run_id": plan.identifier,
            "group": plan.group,
            "model_name": None,
            "commission_name": plan.commission_name,
            "commission": plan.commission,
            "seed": plan.seed,
            "cache_hit": cache_hit,
        }

    submitted = []
    observed_workers = []

    class FakeFuture:
        def __init__(self, payload):
            self.payload = payload

        def result(self):
            return self.payload

    class FakeExecutor:
        def __init__(self, *, max_workers, **kwargs):
            observed_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, plan):
            submitted.append(plan.identifier)
            return FakeFuture(
                {
                    "ok": True,
                    "record": record_for(plan, False),
                    "error_type": "",
                    "error_message": "",
                    "traceback": "",
                    "started_at_utc": "2024-01-01T00:00:00+00:00",
                    "finished_at_utc": "2024-01-01T00:00:01+00:00",
                    "elapsed_seconds": 1.0,
                    "torch_num_threads": 1,
                }
            )

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.plan_training_run", fake_plan
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._cached_record",
        lambda plan, prepared, config: record_for(plan, True),
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.ProcessPoolExecutor", FakeExecutor
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.as_completed", lambda futures: futures
    )

    records = run_training_matrix(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        workers=4,
        show_progress=False,
    )

    assert observed_workers == [4]
    assert submitted == planned_ids[1:]
    assert [record["run_id"] for record in records] == planned_ids
    assert records[0]["cache_hit"] is True
    assert all(record["cache_hit"] is False for record in records[1:])


def test_training_matrix_all_semantic_hits_never_materializes(tmp_path, monkeypatch):
    split = time_split()
    config = experiment_config(split)
    settings = config.resolve_run_settings("smoke")
    prepared = PreparedWindow(
        split=split,
        ticker_group=TICKER_GROUP,
        dataset_id=DATASET_ID,
        tickers=TICKERS,
        real_files=[],
        synthetic_files_by_model={"toy__two_paths": []},
        real_pipeline={},
        synthetic_pipelines={},
        training_data_fingerprints={
            "real_trained": "real",
            "synthetic::toy__two_paths": "synthetic",
            "real_synthetic::toy__two_paths": "combined",
        },
        data_fingerprint="data",
        data_summary=pd.DataFrame(),
        scale_summary=pd.DataFrame(),
    )
    paths = ExperimentPaths(
        artifact_root=tmp_path,
        output_root=tmp_path,
        experiment_root=tmp_path / "experiment",
        train_cache_root=tmp_path / "train_cache",
        evaluation_cache_root=tmp_path / "evaluation_cache",
        experiment_id="all-hit",
    )
    for commission_name, commission in settings.commissions:
        for seed in settings.seeds:
            for group in training_groups(
                prepared.model_names, config.training_group_kinds
            ):
                key, payload = training_cache_key(
                    config=config,
                    settings=settings,
                    prepared=prepared,
                    paths=paths,
                    group=group,
                    commission=commission,
                    seed=seed,
                )
                cache_dir = (
                    paths.train_cache_root
                    / f"semantic_{group.split('::', 1)[0]}__{key[:12]}"
                )
                cache_dir.mkdir(parents=True)
                (cache_dir / "best_model.zip").write_bytes(b"best")
                (cache_dir / "last_model.zip").write_bytes(b"last")
                (cache_dir / "training_log.csv").write_text("reward\n1\n")
                (cache_dir / "validation_log.csv").write_text("reward\n1\n")
                (cache_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "completed": True,
                            "cache_key": key,
                            "semantic_cache_config": payload,
                            "group": group,
                            "run_id": f"{group}-{commission_name}-{seed}",
                        }
                    )
                )

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.resolve_legacy_training_cache_aliases",
        lambda **kwargs: pytest.fail("semantic hit calculated a legacy fingerprint"),
    )
    records = run_training_matrix(
        prepared=prepared,
        config=config,
        settings=settings,
        paths=paths,
        cache_index=TrainingCacheIndex(paths.train_cache_root),
        materialize_prepared=lambda: pytest.fail("cache hit materialized dataframes"),
        workers=1,
        show_progress=False,
    )

    assert len(records) == 6
    assert all(record["cache_hit"] is True for record in records)
    assert not prepared.is_materialized


def test_training_cache_index_scans_status_files_only_once(tmp_path):
    cache_root = tmp_path / "train_cache"
    payloads = [{"schema": "one", "value": 1}, {"schema": "two", "value": 2}]
    keys = [_content_hash(payload) for payload in payloads]
    for index, (key, payload) in enumerate(zip(keys, payloads)):
        cache_dir = cache_root / f"legacy-{index}"
        cache_dir.mkdir(parents=True)
        (cache_dir / "best_model.zip").write_bytes(b"best")
        (cache_dir / "last_model.zip").write_bytes(b"last")
        (cache_dir / "training_log.csv").write_text("reward\n1\n")
        (cache_dir / "validation_log.csv").write_text("reward\n1\n")
        (cache_dir / "status.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "cache_key": key,
                    "semantic_cache_config": payload,
                }
            )
        )
    index = TrainingCacheIndex(cache_root)

    assert index.find(keys[0], cache_root / "preferred-one") is not None
    first_scan_count = index.statistics()["training_cache_statuses_scanned"]
    assert index.find(keys[1], cache_root / "preferred-two") is not None

    assert first_scan_count == 2
    assert index.statistics()["training_cache_statuses_scanned"] == 2


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
    model_path = tmp_path / "model"
    model_path.with_suffix(".zip").write_bytes(b"continuous-model")

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
            "best_model_path": model_path,
            "best_timestep": 4,
            "cache_key": f"training-{split.name}",
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
        evaluation_cache_root=tmp_path / "evaluation_cache",
        show_progress=False,
    )

    np.testing.assert_allclose(observed_initial_weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(observed_initial_weights[1], [0.1, 0.6, 0.3])
    transitions = artifacts["window_transition_log"]
    assert transitions.iloc[1]["initial_portfolio_value"] == 101_000
    assert artifacts["continuous_metrics"].iloc[0]["test_days"] == 2

    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real.rollout",
        lambda *args, **kwargs: pytest.fail("continuous rollout should be cached"),
    )
    cached = evaluate_continuous_test_chains(
        window_results=[window_result(first_split), window_result(second_split)],
        config=config,
        settings=settings,
        evaluation_cache_root=tmp_path / "evaluation_cache",
        show_progress=False,
    )

    assert cached["continuous_evaluation_timings"]["evaluation_cache_hit"].tolist() == [
        True
    ]
    assert cached["continuous_metrics"].iloc[0]["test_days"] == 2


def test_path_count_continuous_evaluation_reuses_legacy_standard_cache(
    tmp_path, monkeypatch
):
    split = time_split()
    config = path_count_experiment_config(
        split, counts=(2,), equivalents={2: "toy__two_paths"}
    ).for_study("path-count")
    settings = config.resolve_run_settings("smoke")
    frame = market_rows(pd.bdate_range("2024-01-01", "2024-02-05"))
    real_files = write_real_files(tmp_path, frame)
    write_synthetic_paths(tmp_path, frame, model_name="toy__two_paths", count=2)
    write_synthetic_paths(tmp_path, frame, model_name="toy__many_paths", count=2)
    prepared = prepare_window_data(
        project_root=tmp_path,
        config=config,
        split=split,
        real_files=real_files,
        real_raw=frame,
        ticker_group=TICKER_GROUP,
        tickers=TICKERS,
    )
    paths = resolve_experiment_paths(
        project_root=tmp_path,
        config=config,
        settings=settings,
        prepared=prepared,
    )
    group = "real_synthetic::toy__many_paths__first_2_paths"
    legacy = legacy_training_cache_key(
        config=config,
        settings=settings,
        prepared=prepared,
        paths=paths,
        group=group,
        commission=0.0025,
        seed=0,
    )
    assert legacy is not None
    model_path = tmp_path / "continuous_legacy_model"
    model_bytes = b"continuous-legacy-model"
    model_path.with_suffix(".zip").write_bytes(model_bytes)
    record = {
        "run_id": "current-continuous-run",
        "group": group,
        "model_name": "toy__many_paths__first_2_paths",
        "commission_name": "with_fee",
        "commission": 0.0025,
        "seed": 0,
        "best_timestep": 2048,
        "best_model_path": model_path,
        "cache_key": legacy[0],
        "training_cache_source_group": "real_synthetic::toy__two_paths",
        "training_cache_source_run_id": "legacy-continuous-run",
    }
    window = WindowResult(
        split=split,
        ticker_group=TICKER_GROUP,
        dataset_id=DATASET_ID,
        tickers=TICKERS,
        experiment_root=paths.experiment_root,
        experiment_id=paths.experiment_id,
        run_table=pd.DataFrame(),
        test_metrics=pd.DataFrame(),
        aggregate_summary=pd.DataFrame(),
        paired_deltas=pd.DataFrame(),
        representative_runs=pd.DataFrame(),
        prepared=prepared,
        run_records=[record],
    )
    legacy_group = "real_synthetic::toy__two_paths"
    legacy_payload = {
        "schema": LEGACY_CONTINUOUS_EVALUATION_CACHE_SCHEMA,
        "ticker_group": TICKER_GROUP,
        "group": legacy_group,
        "commission_name": "with_fee",
        "commission": 0.0025,
        "seed": 0,
        "initial_amount": config.initial_amount,
        "reward_scaling": config.reward_scaling,
        "environment": _evaluation_environment_payload(
            config, 0.0025, cwd=paths.experiment_root
        ),
        "windows": [
            {
                "split": asdict(split),
                "data_fingerprint": prepared.legacy_standard_data_fingerprint,
                "training_cache_key": legacy[0],
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "best_timestep": 2048,
            }
        ],
    }
    legacy_key = _content_hash(legacy_payload)
    identifier = "toy_2__real_synthetic_toy__two_paths__with_fee__seed_0"
    cache_root = tmp_path / "evaluation_cache"
    cache_dir = cache_root / "continuous" / f"{identifier}__{legacy_key[:12]}"
    common = {
        "group": legacy_group,
        "model_name": "toy__two_paths",
        "agent": "Real + Toy",
        "commission_name": "with_fee",
        "commission": 0.0025,
        "seed": 0,
    }
    legacy_frames = {
        "continuous_metrics": pd.DataFrame([{**common, "test_days": 1}]),
        "continuous_window_metrics": pd.DataFrame(
            [{**common, "window_id": split.name, "test_days": 1}]
        ),
        "continuous_daily_returns": pd.DataFrame(
            [{**common, "date": pd.Timestamp(split.test_start), "daily_return": 0.1}]
        ),
        "continuous_equity_curves": pd.DataFrame(
            [{**common, "date": pd.Timestamp(split.test_start), "growth_of_one": 1.1}]
        ),
        "continuous_portfolio_weights": pd.DataFrame(
            [{**common, "date": pd.Timestamp(split.test_start), "asset": "Cash"}]
        ),
        "window_transition_log": pd.DataFrame(
            [{**common, "window_id": split.name, "boundary_turnover": 0.0}]
        ),
    }
    _write_evaluation_cache(
        cache_dir=cache_dir,
        cache_key=legacy_key,
        schema=LEGACY_CONTINUOUS_EVALUATION_CACHE_SCHEMA,
        artifact_names=CONTINUOUS_EVALUATION_ARTIFACTS,
        frames=legacy_frames,
        metadata={"semantic_cache_config": legacy_payload},
    )
    monkeypatch.setattr(
        "finrl.experiments.synthetic_vs_real._evaluate_continuous_chain",
        lambda **kwargs: pytest.fail("legacy continuous cache should be reused"),
    )

    artifacts = evaluate_continuous_test_chains(
        window_results=[window],
        config=config,
        settings=settings,
        evaluation_cache_root=cache_root,
        show_progress=False,
    )

    timings = artifacts["continuous_evaluation_timings"]
    assert timings.iloc[0]["evaluation_cache_hit"]
    assert timings.iloc[0]["evaluation_cache_match_type"] == "legacy_alias"
    assert artifacts["continuous_metrics"].iloc[0]["group"] == group
    assert artifacts["continuous_metrics"].iloc[0]["model_name"] == record["model_name"]
