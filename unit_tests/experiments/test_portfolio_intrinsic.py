from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.experiments.notebook_utils import EIIEExtractor
from finrl.experiments.portfolio_intrinsic import (
    build_portfolio_training_environment,
)
from finrl.experiments.portfolio_intrinsic import (
    plan_portfolio_training,
)
from finrl.experiments.portfolio_intrinsic import portfolio_environment_kwargs
from finrl.experiments.portfolio_intrinsic import portfolio_ppo_kwargs
from finrl.experiments.portfolio_intrinsic import (
    prepare_real_portfolio_periods,
)
from finrl.experiments.portfolio_intrinsic import resolve_variant_reward_config
from finrl.experiments.portfolio_intrinsic import train_portfolio_variant
from finrl.meta.rewards import IntrinsicRewardController
from finrl.meta.rewards import PaperFaithfulIntrinsicRewardController
from finrl.meta.rewards import RobustIntrinsicRewardController


TICKERS = ("AAA", "BBB")


def market_frame(days: int = 30) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=days)
    for day, date in enumerate(dates):
        for ticker_index, ticker in enumerate(TICKERS):
            close = 10.0 + 5.0 * ticker_index + 0.1 * day
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


def tiny_ppo_kwargs() -> dict:
    kwargs = portfolio_ppo_kwargs("cpu")
    kwargs.update(
        {
            "learning_rate": 1e-3,
            "n_steps": 4,
            "batch_size": 4,
            "n_epochs": 1,
        }
    )
    return kwargs


def test_real_data_pipeline_and_ppo_settings_match_control(tmp_path):
    frame = market_frame()
    dates = sorted(frame["date"].unique())
    periods, scaler = prepare_real_portfolio_periods(
        frame,
        TICKERS,
        str(pd.Timestamp(dates[0]).date()),
        str(pd.Timestamp(dates[11]).date()),
        str(pd.Timestamp(dates[11]).date()),
        str(pd.Timestamp(dates[18]).date()),
        str(pd.Timestamp(dates[18]).date()),
        str((pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).date()),
        time_window=3,
    )
    assert set(periods) == {"train", "validation", "test"}
    assert periods["validation"]["date"].nunique() == 3 + 7
    assert periods["test"]["date"].nunique() == 3 + 12
    assert set(scaler.scales) == set(TICKERS)
    assert periods["train"][["close", "high", "low"]].max().max() <= 1.0

    env_kwargs = portfolio_environment_kwargs(0.0025, cwd=tmp_path)
    assert env_kwargs["time_window"] == 50
    assert env_kwargs["reward_scaling"] == 100.0
    assert env_kwargs["action_space_mode"] == "symmetric"
    assert env_kwargs["action_scale"] == 5.0
    assert env_kwargs["return_last_action"] is True

    ppo_kwargs = portfolio_ppo_kwargs("cpu")
    assert ppo_kwargs["learning_rate"] == 3e-4
    assert ppo_kwargs["n_steps"] == 2_048
    assert ppo_kwargs["batch_size"] == 256
    assert ppo_kwargs["n_epochs"] == 10
    assert ppo_kwargs["policy_kwargs"]["log_std_init"] == -2.0
    assert ppo_kwargs["policy_kwargs"]["features_extractor_class"] is EIIEExtractor


def test_portfolio_training_cache_round_trip(tmp_path):
    frame = market_frame(days=12)[["date", "tic", "close", "high", "low"]]
    kwargs = portfolio_environment_kwargs(
        0.0,
        cwd=tmp_path,
        initial_amount=1_000,
        time_window=3,
    )
    arguments = {
        "train": frame,
        "validation": frame,
        "variant": "baseline",
        "commission_name": "no_fee",
        "commission": 0.0,
        "total_timesteps": 8,
        "eval_freq": 4,
        "seed": 0,
        "warmup_steps": 2,
        "env_kwargs": kwargs,
        "ppo_kwargs": tiny_ppo_kwargs(),
        "cache_dir": tmp_path / "cache",
        "replay_capacity": 32,
    }
    first_plan = plan_portfolio_training(**arguments)
    trained = train_portfolio_variant(
        train=frame,
        validation=frame,
        plan=first_plan,
        total_timesteps=8,
        eval_freq=4,
        warmup_steps=2,
        env_kwargs=kwargs,
        ppo_kwargs=arguments["ppo_kwargs"],
        show_progress=False,
        replay_capacity=32,
    )
    cached_plan = plan_portfolio_training(**arguments)
    cached = train_portfolio_variant(
        train=frame,
        validation=frame,
        plan=cached_plan,
        total_timesteps=8,
        eval_freq=4,
        warmup_steps=2,
        env_kwargs=kwargs,
        ppo_kwargs=arguments["ppo_kwargs"],
        show_progress=False,
        replay_capacity=32,
    )

    assert first_plan.cache_hit is False
    assert trained.cache_hit is False
    assert cached_plan.cache_hit is True
    assert cached.cache_hit is True
    assert cached.cache_key == trained.cache_key
    assert not cached.validation_log.empty
    assert {
        "train_reward_mean",
        "real_valid_reward",
        "real_valid_turnover",
        "real_valid_max_weight",
        "is_best",
    }.issubset(cached.validation_log.columns)
    assert np.isfinite(cached.validation_log["real_valid_reward"]).all()
    assert (first_plan.run_dir / "best_model.zip").is_file()
    assert (first_plan.run_dir / "last_model.zip").is_file()


def test_surprise_variants_resolve_distinct_controllers_and_paper_alpha(tmp_path):
    frame = market_frame(days=12)[["date", "tic", "close", "high", "low"]]
    kwargs = portfolio_environment_kwargs(
        0.0025,
        cwd=tmp_path,
        initial_amount=1_000,
        time_window=3,
    )
    paper_config = resolve_variant_reward_config(frame, "paper_faithful", kwargs)
    robust_config = resolve_variant_reward_config(frame, "robust_surprise", kwargs)
    legacy_config = resolve_variant_reward_config(frame, "surprise", kwargs)

    assert paper_config["observation_dim"] == 21
    assert paper_config["effective_alpha"] == 0.05 / 21
    assert robust_config["effective_alpha"] == 0.05
    assert legacy_config["effective_alpha"] == 0.05

    controllers = {}
    for variant in ("paper_faithful", "robust_surprise", "surprise"):
        environment = build_portfolio_training_environment(
            train=frame,
            variant=variant,
            total_timesteps=8,
            seed=0,
            warmup_steps=2,
            env_kwargs=kwargs,
            device="cpu",
            replay_capacity=32,
        )
        controllers[variant] = type(environment.intrinsic_controller)

    assert controllers["paper_faithful"] is PaperFaithfulIntrinsicRewardController
    assert controllers["robust_surprise"] is RobustIntrinsicRewardController
    assert controllers["surprise"] is IntrinsicRewardController
