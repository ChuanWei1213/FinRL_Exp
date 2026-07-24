"""Gymnasium-compatible wrapper around PortfolioOptimizationEnv.

The original environment is built on the legacy `gym` package, so Stable-Baselines3
2.x rejects it (SB3 checks `isinstance(env, gymnasium.Env)`). This subclass inherits
from both, converts the spaces to gymnasium ones and adapts `reset` to the gymnasium
signature. The original module is untouched -- `PolicyGradient` keeps working exactly
as before.

Usage::

    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
        PortfolioOptimizationGymnasiumEnv,
    )
    from stable_baselines3 import PPO

    env = PortfolioOptimizationGymnasiumEnv(df, initial_amount=100000, time_window=50)
    model = PPO("MlpPolicy", env).learn(50_000)
"""
from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from .env_portfolio_optimization import PortfolioOptimizationEnv


def _to_gymnasium_space(space):
    """Rebuild a legacy gym space as the equivalent gymnasium space."""
    if isinstance(space, dict) or space.__class__.__name__ == "Dict":
        return spaces.Dict(
            {k: _to_gymnasium_space(v) for k, v in space.spaces.items()}
        )
    return spaces.Box(
        low=np.asarray(space.low, dtype=np.float32),
        high=np.asarray(space.high, dtype=np.float32),
        shape=space.shape,
        dtype=np.float32,
    )


class PortfolioOptimizationGymnasiumEnv(PortfolioOptimizationEnv, gymnasium.Env):
    """PortfolioOptimizationEnv exposed through the gymnasium API.

    `new_gym_api` is forced to True so that `step` returns the 5-tuple
    (obs, reward, terminated, truncated, info) that gymnasium expects.

    Args:
        action_space_mode: "symmetric" (default) exposes Box(-1, 1) and multiplies
            incoming actions by `action_scale` before handing them to the base
            environment, which then softmaxes them into weights. "legacy" keeps the
            base environment's Box(0, 1) and passes actions through untouched.
        action_scale: Multiplier applied in "symmetric" mode. Ignored in "legacy".

    Note:
        Why "symmetric" is the default: SB3 clips actions to the declared space, so
        under Box(0, 1) the softmax input is confined to [0, 1] and its dynamic range
        is only e**1. With N+1 = 11 assets that caps any single weight at
        e / (e + 10) = 0.21 -- the agent could never concentrate or go fully to cash.
        Scaling to [-action_scale, action_scale] restores the full simplex.
        `PolicyGradient` is unaffected either way: EIIE already emits normalized
        weights, so the base environment uses them as-is.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        *args,
        render_mode=None,
        action_space_mode="symmetric",
        action_scale=5.0,
        plot_on_terminal=False,
        **kwargs,
    ):
        if action_space_mode not in ("symmetric", "legacy"):
            raise ValueError(
                f"action_space_mode must be 'symmetric' or 'legacy', got {action_space_mode!r}"
            )
        kwargs["new_gym_api"] = True
        super().__init__(*args, **kwargs)

        self.render_mode = render_mode
        self.action_space_mode = action_space_mode
        self.action_scale = float(action_scale)
        self.plot_on_terminal = plot_on_terminal

        self.observation_space = _to_gymnasium_space(self.observation_space)
        if action_space_mode == "symmetric":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=self.action_space.shape, dtype=np.float32
            )
        else:
            self.action_space = _to_gymnasium_space(self.action_space)

    def reset(self, seed=None, options=None):
        """Gymnasium-style reset. Returns (observation, info)."""
        gymnasium.Env.reset(self, seed=seed)
        if seed is not None:
            self._seed(seed)
        state, info = super().reset()
        return self._as_float32(state), info

    def step(self, action):
        # The base env renders three matplotlib figures and a quantstats snapshot on
        # the terminal step. Over an RL run that is hundreds of episodes of wasted
        # wall-clock, so short-circuit it with the same return values.
        if not self.plot_on_terminal and self._time_index >= len(self._sorted_times) - 1:
            self._terminal = True
            return self._as_float32(self._state), float(self._reward), True, False, self._info

        if self.action_space_mode == "symmetric":
            action = np.asarray(action, dtype=np.float32) * self.action_scale
        state, reward, terminated, truncated, info = super().step(action)
        return self._as_float32(state), float(reward), bool(terminated), bool(truncated), info

    def max_achievable_weight(self):
        """Largest weight the agent can put on a single asset, given the action space.

        Diagnostic for the softmax dynamic-range issue described in the class docstring.
        """
        n = self.action_space.shape[0]
        hi = self.action_scale if self.action_space_mode == "symmetric" else 1.0
        lo = -hi if self.action_space_mode == "symmetric" else 0.0
        return float(np.exp(hi) / (np.exp(hi) + (n - 1) * np.exp(lo)))

    def render(self):
        return self._state

    @staticmethod
    def _as_float32(state):
        if isinstance(state, dict):
            return {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}
        return np.asarray(state, dtype=np.float32)
