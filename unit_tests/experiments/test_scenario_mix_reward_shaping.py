"""Training-only reward shaping in ScenarioMixEnv (loss weighting and episode tail penalty)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.experiments import scenario_mix as S
from finrl.experiments import scenario_mix_common as common


@pytest.fixture
def market(tmp_path):
    dates = pd.bdate_range("2010-01-01", periods=180)
    rng = np.random.default_rng(5)
    # Volatile enough that episodes contain both losing steps and losing episodes.
    close = 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.02, (180, 10)), axis=0))
    prices = tmp_path / "prices.csv"
    S.episode_frame(dates, close, close * 1.01, close * 0.99, common.TICKERS).to_csv(
        prices, index=False
    )
    return S.MarketData(prices, common.TICKERS), [str(dates[0].date()), str(dates[-1].date())]


def rollout(market, n_episodes=6, **shaping):
    m, (start, end) = market
    env = S.ScenarioMixEnv(m, None, p_synthetic=0.0, data_start=start, data_end=end,
                           commission=0.0025, seed=3, **shaping)
    action_rng = np.random.default_rng(11)
    episodes = []
    for _ in range(n_episodes):
        env.reset()
        steps, done = [], False
        while not done:
            _, reward, terminated, truncated, info = env.step(action_rng.uniform(-1, 1, 11))
            done = terminated or truncated
            steps.append((reward, info))
        episodes.append(steps)
    return episodes


def test_default_reward_is_unchanged_log_return(market):
    for steps in rollout(market):
        rewards = np.array([r for r, _ in steps])
        final = steps[-1][1]
        # Non-terminal steps pass the env's info through untouched.
        assert all("reward_base" not in info for _, info in steps[:-1])
        assert final["reward_base"] == rewards[-1]
        assert final["reward_loss_term"] == 0.0 and final["reward_tail_term"] == 0.0
        # Episode return is consistent with the summed log-return reward (scale 100).
        np.testing.assert_allclose(final["episode_return"], np.expm1(rewards.sum() / 100), rtol=1e-9)


def test_shaping_off_matches_original_reward_stream(market):
    base = rollout(market)
    zero = rollout(market, loss_weight=0.0, tail_weight=0.0)
    for a, b in zip(base, zero):
        assert [r for r, _ in a] == [r for r, _ in b]


def test_loss_weight_scales_only_negative_steps(market):
    base = rollout(market)
    shaped = rollout(market, loss_weight=1.0)
    seen_negative = False
    for a, b in zip(base, shaped):
        for (r0, _), (r1, info) in zip(a, b):
            assert info["reward_base"] == pytest.approx(r0)
            expected = 2 * r0 if r0 < 0 else r0
            assert r1 == pytest.approx(expected)
            seen_negative |= r0 < 0
    assert seen_negative


def test_tail_penalty_only_on_terminal_step_below_threshold(market):
    base = rollout(market, n_episodes=12)
    shaped = rollout(market, n_episodes=12, tail_weight=2.0, tail_threshold=-0.02)
    penalized = 0
    for a, b in zip(base, shaped):
        for (r0, _), (r1, _) in zip(a[:-1], b[:-1]):
            assert r1 == pytest.approx(r0)
        info = b[-1][1]
        shortfall = max(0.0, -0.02 - info["episode_return"])
        assert info["reward_tail_term"] == pytest.approx(-100 * 2.0 * shortfall)
        assert b[-1][0] == pytest.approx(a[-1][0] + info["reward_tail_term"])
        penalized += shortfall > 0
    assert penalized > 0


@pytest.mark.parametrize("kwargs", [{"loss_weight": -1}, {"tail_weight": -0.1},
                                    {"tail_threshold": 0.1}, {"tail_threshold": -1.0}])
def test_invalid_shaping_arguments(market, kwargs):
    m, (start, end) = market
    with pytest.raises(ValueError):
        S.ScenarioMixEnv(m, None, p_synthetic=0.0, data_start=start, data_end=end,
                         commission=0.0025, **kwargs)


@pytest.mark.parametrize("shape,g", [
    ({"loss_power": 2.0}, lambda x: x ** 2),
    ({"loss_power": 3.0}, lambda x: x ** 3),
    ({"loss_exp_rate": 0.5}, lambda x: np.expm1(0.5 * x) / 0.5),
])
def test_convex_loss_shapes(market, shape, g):
    base = rollout(market)
    shaped = rollout(market, loss_weight=0.7, **shape)
    for a, b in zip(base, shaped):
        for (r0, _), (r1, _) in zip(a, b):
            assert r1 == pytest.approx(r0 - 0.7 * g(max(-r0, 0.0)))


def test_convex_power_spares_small_losses_and_weighs_large_ones(market):
    m, (start, end) = market
    env = S.ScenarioMixEnv(m, None, p_synthetic=0.0, data_start=start, data_end=end,
                           commission=0.0025, loss_weight=1.0, loss_power=2.0)
    assert env._loss_penalty(0.3) == pytest.approx(0.09)     # below linear 0.3
    assert env._loss_penalty(1.0) == pytest.approx(1.0)      # equal at 1% (reward unit)
    assert env._loss_penalty(6.0) == pytest.approx(36.0)     # far above linear 6


def test_loss_cap_limits_per_step_penalty(market):
    base = rollout(market)
    shaped = rollout(market, loss_weight=1.0, loss_power=3.0, loss_cap=5.0)
    for a, b in zip(base, shaped):
        for (r0, _), (r1, info) in zip(a, b):
            assert r1 == pytest.approx(r0 - min(max(-r0, 0.0) ** 3, 5.0))


@pytest.mark.parametrize("kwargs", [{"loss_power": 0.5}, {"loss_exp_rate": -1},
                                    {"loss_power": 2.0, "loss_exp_rate": 0.5},
                                    {"loss_cap": 0.0}])
def test_invalid_loss_shape_arguments(market, kwargs):
    m, (start, end) = market
    with pytest.raises(ValueError):
        S.ScenarioMixEnv(m, None, p_synthetic=0.0, data_start=start, data_end=end,
                         commission=0.0025, loss_weight=1.0, **kwargs)
