"""Check that the comparison isolates fees and reproduces the official evaluator."""
import numpy as np
import pandas as pd
import pytest

from analysis.scenario_mix.trf_ablation import paired_rollout, trf_version
from finrl.experiments import scenario_mix as S
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv


def market():
    dates = pd.bdate_range("2020-01-01", periods=90)
    t = np.arange(90)[:, None]
    prices = 100 * np.exp(t * np.array([.003, -.002])[None, :])
    return S.MarketData(S.episode_frame(dates, prices, prices*1.01, prices*.99, ["AAA", "BBB"]))


class Policy:
    def predict(self, obs, deterministic=True):
        return np.array([-.1, .2, -.3], np.float32), None


def test_paired_replay_matches_official_and_keeps_policy_weights():
    m = market()
    start, end = str(m.dates[55].date()), str(m.dates[-1].date())
    paired = paired_rollout(Policy(), m, start, end)
    ref = S.run_continuous(lambda obs: Policy().predict(obs)[0], m, start, end, .0025)
    pd.testing.assert_frame_equal(paired["fixed"], ref)
    np.testing.assert_allclose(paired["fixed"].cash_weight, paired["legacy"].cash_weight, atol=1e-7)
    assert not np.array_equal(paired["fixed"].value, paired["legacy"].value)


def test_zero_cost_outputs_are_identical():
    m = market()
    paired = paired_rollout(Policy(), m, str(m.dates[55].date()), str(m.dates[-1].date()), 0)
    pd.testing.assert_frame_equal(paired["fixed"], paired["legacy"])


def test_legacy_patch_restores_after_exception():
    original = PortfolioOptimizationEnv.step
    with pytest.raises(RuntimeError, match="intentional"):
        with trf_version("legacy"):
            assert PortfolioOptimizationEnv.step is not original
            raise RuntimeError("intentional")
    assert PortfolioOptimizationEnv.step is original


@pytest.mark.parametrize("variant,expected", [("fixed", .9975), ("legacy", 1.0)])
def test_known_cash_purchase_uses_requested_version(variant, expected):
    m = market()
    frame = S.episode_frame(m.dates[:4], np.full((4,2),100.), np.full((4,2),100.),
                            np.full((4,2),100.), m.tickers)
    with trf_version(variant):
        env = S._fast_env_class()(frame, **S.env_kwargs(.0025, obs_days=1))
        env.reset()
        _, _, _, _, info = env.step([0, .2, 0])  # adapter scales to exact [0,1,0]
    assert info["trf_mu"] == pytest.approx(expected, abs=1e-9)
