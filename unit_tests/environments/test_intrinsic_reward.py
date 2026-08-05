from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_intrinsic import (
    IntrinsicRewardStockTradingEnv,
)
from finrl.meta.rewards import IntrinsicRewardConfig
from finrl.meta.rewards import IntrinsicRewardController
from finrl.meta.rewards import StableSurpriseModel


TICKERS = ("AAA", "BBB")
INDICATORS = ("feature",)


def make_market_data(days: int = 8) -> pd.DataFrame:
    rows = []
    index = []
    for day in range(days):
        for ticker_index, ticker in enumerate(TICKERS):
            rows.append(
                {
                    "date": f"2024-01-{day + 1:02d}",
                    "tic": ticker,
                    "close": 100.0 + 25.0 * ticker_index + day * (ticker_index + 1),
                    "feature": 0.1 + 0.01 * day + 0.001 * ticker_index,
                }
            )
            index.append(day)
    dataframe = pd.DataFrame(rows)
    dataframe.index = index
    return dataframe


def environment_kwargs() -> dict:
    stock_dimension = len(TICKERS)
    return {
        "stock_dim": stock_dimension,
        "hmax": 10,
        "initial_amount": 100_000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "reward_scaling": 1e-4,
        "state_space": 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension,
        "action_space": stock_dimension,
        "tech_indicator_list": list(INDICATORS),
        "print_verbosity": 10_000,
    }


def intrinsic_config(
    *,
    alpha: float = 0.05,
    beta: float = 0.05,
    warmup_steps: int = 0,
    total_timesteps: int = 32,
) -> IntrinsicRewardConfig:
    return IntrinsicRewardConfig(
        total_timesteps=total_timesteps,
        alpha=alpha,
        beta=beta,
        warmup_steps=warmup_steps,
        batch_size=2,
        replay_capacity=32,
        latent_dim=2,
        volatility_window=3,
        seed=7,
        device="cpu",
    )


def test_surprise_model_matches_manual_gaussian_nll_and_stays_finite():
    model = StableSurpriseModel(
        observation_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
        std_min=0.05,
        std_max=5.0,
    )
    with torch.no_grad():
        for parameter in model.predictor.parameters():
            parameter.zero_()
        model.mean_head.weight.zero_()
        model.mean_head.bias.zero_()
        model.std_head[0].weight.zero_()
        desired_softplus = 1.0 - model.std_min
        inverse_softplus = math.log(math.expm1(desired_softplus))
        model.std_head[0].bias.fill_(inverse_softplus)

    bonus = model.compute_bonus(
        np.zeros(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
    )
    expected = 0.5 * (math.log(2.0 * math.pi) + 1.0)
    assert bonus == pytest.approx(expected, rel=1e-5)

    extreme_bonus = model.compute_bonus(
        np.full(2, 1e6, dtype=np.float32),
        np.full(1, 1e6, dtype=np.float32),
        np.full(2, -1e6, dtype=np.float32),
    )
    assert math.isfinite(extreme_bonus)


def test_controller_warmup_scaling_decay_and_reward_formula():
    config = intrinsic_config(warmup_steps=2, total_timesteps=10)
    controller = IntrinsicRewardController(3, 1, config)
    controller.observe_initial(np.zeros(3, dtype=np.float32))
    portfolio_values = [100.0, 102.0, 99.0, 104.0]

    first_reward, first_info = controller.process_transition(
        np.zeros(3), np.zeros(1), np.ones(3), 0.25, portfolio_values
    )
    second_reward, second_info = controller.process_transition(
        np.ones(3), np.ones(1), np.full(3, 2.0), 0.25, portfolio_values
    )
    third_reward, third_info = controller.process_transition(
        np.full(3, 2.0), np.zeros(1), np.full(3, 3.0), 0.25, portfolio_values
    )

    assert first_reward == pytest.approx(0.25)
    assert second_reward == pytest.approx(0.25)
    assert first_info["intrinsic_warmup"] is True
    assert second_info["intrinsic_warmup"] is True
    assert third_info["intrinsic_warmup"] is False
    expected_intrinsic = third_info["intrinsic_eta"] * (
        config.alpha * third_info["reward_surprise"]
        + config.beta * third_info["reward_dejavu"]
    )
    assert third_info["reward_intrinsic"] == pytest.approx(expected_intrinsic)
    assert third_reward == pytest.approx(0.25 + expected_intrinsic)
    assert 0 <= third_info["reward_surprise"] <= config.bonus_clip
    assert 0 <= third_info["reward_dejavu"] <= config.bonus_clip
    assert first_info["intrinsic_time_decay"] > third_info["intrinsic_time_decay"]


def test_portfolio_volatility_factor_is_bounded_and_monotonic():
    no_history = IntrinsicRewardController.portfolio_volatility_factor(
        [100.0, 101.0], window=20, scale=100.0
    )
    low = IntrinsicRewardController.portfolio_volatility_factor(
        [100.0, 100.2, 100.1, 100.3], window=20, scale=100.0
    )
    high = IntrinsicRewardController.portfolio_volatility_factor(
        [100.0, 110.0, 90.0, 115.0], window=20, scale=100.0
    )
    assert no_history == 0.0
    assert 0 <= low < high < 1


def test_eval_mode_matches_original_environment():
    data = make_market_data()
    kwargs = environment_kwargs()
    base_environment = StockTradingEnv(df=data.copy(), **kwargs)
    intrinsic_environment = IntrinsicRewardStockTradingEnv(
        df=data.copy(),
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="eval",
        **kwargs,
    )
    base_observation, _ = base_environment.reset()
    intrinsic_observation, _ = intrinsic_environment.reset()
    np.testing.assert_allclose(base_observation, intrinsic_observation)

    actions = (
        np.array([0.5, 0.0], dtype=np.float32),
        np.array([0.0, 0.5], dtype=np.float32),
        np.array([-0.25, 0.0], dtype=np.float32),
    )
    for action in actions:
        base_result = base_environment.step(action.copy())
        intrinsic_result = intrinsic_environment.step(action.copy())
        np.testing.assert_allclose(base_result[0], intrinsic_result[0])
        assert base_result[1] == pytest.approx(intrinsic_result[1])
        assert base_result[2:4] == intrinsic_result[2:4]
        assert intrinsic_result[4]["reward_intrinsic"] == 0.0
        assert intrinsic_result[4]["reward_ext"] == pytest.approx(base_result[1])
        assert base_environment.asset_memory == intrinsic_environment.asset_memory
        assert base_environment.cost == pytest.approx(intrinsic_environment.cost)


def test_train_mode_logs_finite_components_and_avoids_terminal_duplicates():
    data = make_market_data(days=5)
    environment = IntrinsicRewardStockTradingEnv(
        df=data,
        intrinsic_config=intrinsic_config(),
        intrinsic_mode="train",
        **environment_kwargs(),
    )
    environment.reset()

    terminated = False
    last_info = None
    while not terminated:
        _, reward, terminated, _, last_info = environment.step(
            np.array([0.5, 0.0], dtype=np.float32)
        )
        assert math.isfinite(float(reward))

    assert last_info is not None
    expected_transitions = len(data.index.unique()) - 1
    assert len(environment.intrinsic_controller.replay_pool) == expected_transitions
    assert environment.intrinsic_controller.global_step == expected_transitions
    assert len(environment.intrinsic_rewards_memory) == expected_transitions
    for key, value in last_info.items():
        if key.startswith("reward_") or key.startswith("intrinsic_"):
            assert math.isfinite(float(value))

    replay_size = len(environment.intrinsic_controller.replay_pool)
    global_step = environment.intrinsic_controller.global_step
    environment.reset()
    assert len(environment.intrinsic_controller.replay_pool) == replay_size
    assert environment.intrinsic_controller.global_step == global_step
    assert environment.intrinsic_rewards_memory == []


def test_controller_checkpoint_round_trip(tmp_path):
    config = intrinsic_config()
    source = IntrinsicRewardController(3, 1, config)
    source.observe_initial(np.zeros(3, dtype=np.float32))
    for index in range(3):
        source.process_transition(
            np.full(3, index, dtype=np.float32),
            np.array([index % 2], dtype=np.float32),
            np.full(3, index + 1, dtype=np.float32),
            0.1,
            [100.0, 101.0, 99.0, 102.0],
        )

    checkpoint = tmp_path / "intrinsic.pt"
    source.save(checkpoint)
    restored = IntrinsicRewardController(3, 1, config)
    restored.load(checkpoint)

    assert restored.global_step == source.global_step
    assert len(restored.replay_pool) == len(source.replay_pool)
    np.testing.assert_allclose(
        restored.observation_stats.mean, source.observation_stats.mean
    )
    np.testing.assert_allclose(
        restored.replay_pool._observations, source.replay_pool._observations
    )
    for source_parameter, restored_parameter in zip(
        source.surprise_model.predictor.parameters(),
        restored.surprise_model.predictor.parameters(),
    ):
        torch.testing.assert_close(source_parameter, restored_parameter)


@pytest.mark.parametrize("variant", ["baseline", "surprise", "dejavu", "combined"])
def test_ppo_variants_smoke_train_and_save(tmp_path, variant):
    data = make_market_data(days=6)
    kwargs = environment_kwargs()
    if variant == "baseline":
        environment = StockTradingEnv(df=data, **kwargs)
    else:
        alpha = 0.05 if variant in {"surprise", "combined"} else 0.0
        beta = 0.05 if variant in {"dejavu", "combined"} else 0.0
        environment = IntrinsicRewardStockTradingEnv(
            df=data,
            intrinsic_config=intrinsic_config(alpha=alpha, beta=beta),
            intrinsic_mode="train",
            **kwargs,
        )

    vector_environment, _ = environment.get_sb_env()
    model = PPO(
        "MlpPolicy",
        vector_environment,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-3,
        seed=3,
        verbose=0,
    )
    model.learn(total_timesteps=8)
    model_path = tmp_path / f"ppo_{variant}"
    model.save(model_path)
    assert model_path.with_suffix(".zip").exists()

    if isinstance(environment, IntrinsicRewardStockTradingEnv):
        checkpoint = tmp_path / f"intrinsic_{variant}.pt"
        environment.save_intrinsic_state(checkpoint)
        assert checkpoint.exists()
