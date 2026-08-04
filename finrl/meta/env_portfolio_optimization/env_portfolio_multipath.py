"""Train across many price paths instead of replaying one history.

The plain environment replays a single price series in fixed order, so at 300k
timesteps the agent sees the same ~1940 (state, outcome) pairs about 155 times. Each
50-day window is effectively a unique key, and memorizing beats generalizing.

`MultiPathEnv` draws a different path per episode, so no single day can be memorized --
the agent has to learn what a *pattern* implies. Feed it paths from a generative model
(diffusion, GAN, bootstrap) fitted on the training years only.

Usage::

    dfs = [real_train_df] + [synth_df_i for i in range(50)]
    env = MultiPathEnv.from_dataframes(dfs, **ENV_KWARGS)
    balanced_env = MultiPathEnv.from_balanced_real_and_synthetic_dataframes(
        real_train_df, synth_dfs, **ENV_KWARGS
    )
    model = PPO("MultiInputPolicy", Monitor(balanced_env), **PPO_KWARGS)

Validation and test must stay on real data -- otherwise you are only measuring how well
the agent fits the generator's distribution.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from .env_portfolio_optimization_gymnasium import PortfolioOptimizationGymnasiumEnv


class MultiPathEnv(gymnasium.Env):
    """Samples a different underlying environment each episode.

    Args:
        envs: Environments to draw from. They must agree on observation and action
            spaces, i.e. same asset count, features and time window.
        seed: Seed for the path sampler (independent of the env seeds).
        weights: Optional sampling probabilities, one per env. Use this to keep the
            real path over-represented, e.g. `[0.2] + [0.8 / n] * n`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, envs, seed=0, weights=None):
        if not envs:
            raise ValueError("need at least one environment")

        ref = envs[0]
        for i, e in enumerate(envs[1:], start=1):
            if e.observation_space != ref.observation_space:
                raise ValueError(
                    f"env {i} observation space {e.observation_space} != {ref.observation_space}"
                )
            if e.action_space != ref.action_space:
                raise ValueError(f"env {i} action space differs from env 0")

        self.envs = list(envs)
        self.observation_space = ref.observation_space
        self.action_space = ref.action_space

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if len(w) != len(envs):
                raise ValueError(f"weights has {len(w)} entries for {len(envs)} envs")
            weights = w / w.sum()
        self.weights = weights

        self._rng = np.random.default_rng(seed)
        self.env = ref
        self.path_index = 0
        self.path_counts = np.zeros(len(envs), dtype=int)

    @classmethod
    def from_dataframes(cls, dfs, seed=0, weights=None, **env_kwargs):
        """Build one PortfolioOptimizationGymnasiumEnv per dataframe."""
        envs = [PortfolioOptimizationGymnasiumEnv(df, **env_kwargs) for df in dfs]
        return cls(envs, seed=seed, weights=weights)

    @classmethod
    def from_balanced_real_and_synthetic_dataframes(
        cls, real_df, synthetic_dfs, seed=0, **env_kwargs
    ):
        """Build an env that samples real and synthetic sources equally.

        The real dataframe receives half of the episode probability. The other
        half is divided uniformly among the existing synthetic dataframes, so
        balancing does not require copying the real dataframe once per synthetic
        path.
        """
        synthetic_dfs = list(synthetic_dfs)
        if not synthetic_dfs:
            raise ValueError("need at least one synthetic dataframe")

        weights = np.full(len(synthetic_dfs) + 1, 0.5 / len(synthetic_dfs))
        weights[0] = 0.5
        return cls.from_dataframes(
            [real_df, *synthetic_dfs],
            seed=seed,
            weights=weights,
            **env_kwargs,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.path_index = int(self._rng.choice(len(self.envs), p=self.weights))
        self.path_counts[self.path_index] += 1
        self.env = self.envs[self.path_index]
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "path_index": self.path_index}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    # the evaluation helpers reach into these; delegate to the live sub-environment
    @property
    def _asset_memory(self):
        return self.env._asset_memory

    @property
    def _actions_memory(self):
        return self.env._actions_memory

    @property
    def _final_weights(self):
        return self.env._final_weights

    @property
    def _reward_scaling(self):
        return self.env._reward_scaling

    @property
    def action_space_mode(self):
        return self.env.action_space_mode
