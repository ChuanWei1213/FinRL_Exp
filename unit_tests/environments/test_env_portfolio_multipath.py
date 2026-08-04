import numpy as np
import pytest

from finrl.meta.env_portfolio_optimization.env_portfolio_multipath import MultiPathEnv


class RecordingMultiPathEnv(MultiPathEnv):
    @classmethod
    def from_dataframes(cls, dfs, seed=0, weights=None, **env_kwargs):
        return {
            "dfs": dfs,
            "seed": seed,
            "weights": weights,
            "env_kwargs": env_kwargs,
        }


def test_balanced_real_and_synthetic_sampling_reuses_dataframes():
    real = object()
    synthetic = [object(), object(), object()]

    result = RecordingMultiPathEnv.from_balanced_real_and_synthetic_dataframes(
        real,
        synthetic,
        seed=17,
        time_window=50,
    )

    assert result["dfs"][0] is real
    assert all(
        actual is expected for actual, expected in zip(result["dfs"][1:], synthetic)
    )
    assert result["seed"] == 17
    assert result["env_kwargs"] == {"time_window": 50}
    np.testing.assert_allclose(result["weights"], [0.5, 1 / 6, 1 / 6, 1 / 6])


def test_balanced_real_and_synthetic_sampling_requires_synthetic_data():
    with pytest.raises(ValueError, match="at least one synthetic dataframe"):
        RecordingMultiPathEnv.from_balanced_real_and_synthetic_dataframes(object(), [])
