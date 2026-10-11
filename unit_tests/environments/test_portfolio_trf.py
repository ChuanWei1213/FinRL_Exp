"""TRF regression checks against trades with known cash flows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.experiments.scenario_mix import _fast_env_class
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
    PortfolioOptimizationGymnasiumEnv,
)


def make_env(tmp_path, commission=0.0025, fast=False, rising=False):
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=5)):
        for ticker in ("AAA", "BBB"):
            price = 100.0 * (2 if rising and day > 0 and ticker == "AAA" else 1)
            rows.append(dict(date=date, tic=ticker, close=price, high=price, low=price))
    cls = _fast_env_class() if fast else PortfolioOptimizationGymnasiumEnv
    env = cls(
        pd.DataFrame(rows),
        initial_amount=1000,
        time_window=1,
        return_last_action=True,
        action_space_mode="legacy",
        comission_fee_pct=commission,
        cwd=tmp_path,
    )
    env.reset()
    return env


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(
    "previous,target,expected",
    [
        ([1, 0, 0], [0, 1, 0], 0.9975),  # buy with all cash
        ([0, 1, 0], [1, 0, 0], 0.9975),  # liquidate
        ([0, 1, 0], [0, 0, 1], 0.9975**2),  # sell, then buy another asset
        ([1, 0, 0], [1, 0, 0], 1.0),
        ([0.25, 0.25, 0.5], [0.25, 0.25, 0.5], 1.0),
        ([1, 0, 0], [0.5, 0.5, 0], 0.9975 / 0.99875),
        ([0, 1, 0], [0.5, 0.5, 0], 0.9975 / 0.99875),
    ],
)
def test_trf_known_trades(tmp_path, fast, previous, target, expected):
    env = make_env(tmp_path, fast=fast)
    env._final_weights[-1] = np.asarray(previous, dtype=np.float32)
    _, reward, _, _, info = env.step(target)
    assert info["trf_mu"] == pytest.approx(expected, abs=1e-9)
    assert env._portfolio_value == pytest.approx(1000 * expected, rel=1e-7)
    assert reward == pytest.approx(np.log(expected), abs=1e-7)


@pytest.mark.parametrize("commission", [0.0, 0.0025, 0.01])
def test_trf_preserves_cash_balance_for_mixed_trades(tmp_path, commission):
    env = make_env(tmp_path, commission=commission)
    rng = np.random.default_rng(17)
    for _ in range(20):
        previous, target = rng.dirichlet(np.ones(3), size=2).astype(np.float32)
        env.reset()
        env._final_weights[-1] = previous
        _, _, _, _, info = env.step(target)
        mu = info["trf_mu"]
        sold = np.maximum(previous[1:] - mu * target[1:], 0).sum()
        bought = np.maximum(mu * target[1:] - previous[1:], 0).sum()
        # Jiang's TRF convention: purchase fees are withheld from cash spent.
        # This cash-flow check does not repeat the fixed-point recurrence.
        cash_after = previous[0] + (1 - commission) * sold - bought / (1 - commission)
        assert cash_after == pytest.approx(mu * target[0], abs=2e-7)
        assert 0 < mu <= 1 + 1e-7


def test_trf_uses_drifted_weights_before_rebalancing(tmp_path):
    env = make_env(tmp_path, rising=True)
    env.step([0, 0.5, 0.5])
    drifted = env._final_weights[-1].copy()
    np.testing.assert_allclose(drifted, [0, 2 / 3, 1 / 3], rtol=1e-7)
    _, _, _, _, info = env.step(drifted)
    assert info["trf_mu"] == pytest.approx(1.0, abs=1e-9)


def test_zero_commission_matches_disabled_fees(tmp_path):
    free = make_env(tmp_path / "free", commission=0, rising=True)
    disabled = make_env(tmp_path / "disabled", commission=0, rising=True)
    disabled._comission_fee_model = None
    for action in ([0.2, 0.3, 0.5], [1, 0, 0], [0, 1, 0]):
        left, right = free.step(action), disabled.step(action)
        assert left[4]["trf_mu"] == 1.0
        assert left[1] == right[1]
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        assert free._portfolio_value == disabled._portfolio_value
