from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from finrl.experiments.portfolio_intrinsic import (
    build_portfolio_training_environment,
)
from finrl.experiments.portfolio_intrinsic import portfolio_environment_kwargs
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
    PortfolioOptimizationGymnasiumEnv,
)
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_intrinsic import (
    flatten_portfolio_observation,
)
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_intrinsic import (
    IntrinsicRewardPortfolioOptimizationEnv,
)
from finrl.meta.rewards import IntrinsicRewardConfig


TICKERS = ("AAA", "BBB")


def make_portfolio_data(days: int = 9) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-02", periods=days)):
        for ticker_index, ticker in enumerate(TICKERS):
            close = 1.0 + 0.05 * ticker_index + 0.01 * day * (ticker_index + 1)
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                }
            )
    return pd.DataFrame(rows)


def environment_kwargs(tmp_path, commission: float = 0.0025) -> dict:
    return portfolio_environment_kwargs(
        commission,
        cwd=tmp_path,
        initial_amount=1_000,
        time_window=3,
        reward_scaling=100.0,
    )


def intrinsic_config() -> IntrinsicRewardConfig:
    return IntrinsicRewardConfig(
        total_timesteps=32,
        alpha=0.05,
        beta=0.05,
        warmup_steps=0,
        batch_size=2,
        replay_capacity=32,
        latent_dim=2,
        volatility_window=3,
        seed=7,
        device="cpu",
    )


def assert_observation_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key])


def test_flatten_portfolio_observation_uses_state_then_last_action():
    observation = {
        "state": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "last_action": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    flattened = flatten_portfolio_observation(observation)
    np.testing.assert_allclose(
        flattened,
        np.concatenate([observation["state"].reshape(-1), observation["last_action"]]),
    )


def test_eval_mode_is_identical_to_original_portfolio_environment(tmp_path):
    data = make_portfolio_data()
    kwargs = environment_kwargs(tmp_path)
    original = PortfolioOptimizationGymnasiumEnv(data.copy(), **kwargs)
    intrinsic = IntrinsicRewardPortfolioOptimizationEnv(
        data.copy(),
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="eval",
        **kwargs,
    )
    original_observation, _ = original.reset(seed=5)
    intrinsic_observation, _ = intrinsic.reset(seed=5)
    assert_observation_equal(original_observation, intrinsic_observation)

    actions = (
        np.array([0.1, 0.4, -0.2], dtype=np.float32),
        np.array([-0.3, 0.2, 0.5], dtype=np.float32),
        np.array([0.0, -0.4, 0.3], dtype=np.float32),
    )
    for action in actions:
        original_result = original.step(action.copy())
        intrinsic_result = intrinsic.step(action.copy())
        assert_observation_equal(original_result[0], intrinsic_result[0])
        assert original_result[1] == pytest.approx(intrinsic_result[1])
        assert original_result[2:4] == intrinsic_result[2:4]
        assert intrinsic_result[4]["reward_ext"] == pytest.approx(original_result[1])
        assert intrinsic_result[4]["reward_intrinsic"] == 0.0
        np.testing.assert_allclose(
            original._asset_memory["final"], intrinsic._asset_memory["final"]
        )
        np.testing.assert_allclose(original._actions_memory, intrinsic._actions_memory)


def test_train_mode_only_changes_returned_reward_and_skips_terminal_duplicate(
    tmp_path,
):
    data = make_portfolio_data(days=8)
    kwargs = environment_kwargs(tmp_path, commission=0.0)
    original = PortfolioOptimizationGymnasiumEnv(data.copy(), **kwargs)
    intrinsic = IntrinsicRewardPortfolioOptimizationEnv(
        data.copy(),
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="train",
        **kwargs,
    )
    original_observation, _ = original.reset(seed=3)
    intrinsic_observation, _ = intrinsic.reset(seed=3)
    assert_observation_equal(original_observation, intrinsic_observation)

    terminated = False
    while not terminated:
        action = np.array([-0.2, 0.4, 0.1], dtype=np.float32)
        original_result = original.step(action.copy())
        intrinsic_result = intrinsic.step(action.copy())
        terminated = intrinsic_result[2]
        assert original_result[2:4] == intrinsic_result[2:4]
        assert_observation_equal(original_result[0], intrinsic_result[0])
        assert intrinsic_result[4]["reward_ext"] == pytest.approx(original_result[1])
        assert intrinsic_result[1] == pytest.approx(
            intrinsic_result[4]["reward_ext"] + intrinsic_result[4]["reward_intrinsic"]
        )
        assert all(
            math.isfinite(float(value))
            for key, value in intrinsic_result[4].items()
            if key.startswith("reward_") or key.startswith("intrinsic_")
        )

    expected_transitions = len(data["date"].unique()) - kwargs["time_window"]
    controller = intrinsic.intrinsic_controller
    assert controller.global_step == expected_transitions
    assert len(controller.replay_pool) == expected_transitions
    assert len(intrinsic.intrinsic_rewards_memory) == expected_transitions
    assert any(value > 0 for value in intrinsic.intrinsic_rewards_memory[2:])
    np.testing.assert_allclose(
        original._asset_memory["final"], intrinsic._asset_memory["final"]
    )
    replay_size = len(controller.replay_pool)
    intrinsic.reset()
    assert len(controller.replay_pool) == replay_size
    assert controller.global_step == expected_transitions
    assert intrinsic.intrinsic_rewards_memory == []


def test_portfolio_intrinsic_checkpoint_round_trip(tmp_path):
    data = make_portfolio_data(days=8)
    kwargs = environment_kwargs(tmp_path, commission=0.0)
    source = IntrinsicRewardPortfolioOptimizationEnv(
        data.copy(),
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="train",
        **kwargs,
    )
    source.reset(seed=9)
    for _ in range(3):
        source.step(np.array([-0.2, 0.4, 0.1], dtype=np.float32))

    checkpoint = tmp_path / "portfolio_intrinsic.pt"
    source.save_intrinsic_state(checkpoint)
    restored = IntrinsicRewardPortfolioOptimizationEnv(
        data.copy(),
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="train",
        **kwargs,
    )
    restored.load_intrinsic_state(checkpoint)

    assert restored.intrinsic_controller.global_step == 3
    assert len(restored.intrinsic_controller.replay_pool) == 3
    np.testing.assert_allclose(
        restored.intrinsic_controller.observation_stats.mean,
        source.intrinsic_controller.observation_stats.mean,
    )


def test_baseline_builder_returns_the_unmodified_original_environment(tmp_path):
    environment = build_portfolio_training_environment(
        train=make_portfolio_data(),
        variant="baseline",
        total_timesteps=32,
        seed=0,
        warmup_steps=2,
        env_kwargs=environment_kwargs(tmp_path),
        device="cpu",
        replay_capacity=32,
    )
    assert type(environment) is PortfolioOptimizationGymnasiumEnv
