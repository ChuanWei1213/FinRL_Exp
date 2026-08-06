from __future__ import annotations

from pathlib import Path
from typing import Literal, Type

import numpy as np

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization_gymnasium import (
    PortfolioOptimizationGymnasiumEnv,
)
from finrl.meta.rewards import IntrinsicRewardConfig
from finrl.meta.rewards import IntrinsicRewardController


IntrinsicMode = Literal["train", "eval"]


def flatten_portfolio_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the EIIE state and previous weights in a stable order."""
    if not isinstance(observation, dict):
        raise TypeError("portfolio intrinsic observations must be dictionaries")
    missing = {"state", "last_action"} - set(observation)
    if missing:
        raise ValueError(f"portfolio observation is missing {sorted(missing)}")
    flattened = np.concatenate(
        [
            np.asarray(observation["state"], dtype=np.float32).reshape(-1),
            np.asarray(observation["last_action"], dtype=np.float32).reshape(-1),
        ]
    )
    if not np.all(np.isfinite(flattened)):
        raise ValueError("portfolio observation contains NaN or infinity")
    return flattened


class IntrinsicRewardPortfolioOptimizationEnv(PortfolioOptimizationGymnasiumEnv):
    """PortfolioOptimizationGymnasiumEnv with a training-only novelty bonus.

    The parent environment remains responsible for action scaling, softmax portfolio
    weights, transaction costs, asset accounting, and the scaled log-return reward.
    This subclass only replaces the reward returned to the training agent.
    """

    def __init__(
        self,
        *args,
        intrinsic_config: IntrinsicRewardConfig,
        intrinsic_mode: IntrinsicMode = "train",
        intrinsic_controller_class: Type[IntrinsicRewardController] = (
            IntrinsicRewardController
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._validate_intrinsic_mode(intrinsic_mode)
        if not self._return_last_action:
            raise ValueError(
                "IntrinsicRewardPortfolioOptimizationEnv requires "
                "return_last_action=True"
            )
        self.intrinsic_mode: IntrinsicMode = intrinsic_mode
        observation_dim = int(
            np.prod(self.observation_space["state"].shape)
            + np.prod(self.observation_space["last_action"].shape)
        )
        self.intrinsic_controller = intrinsic_controller_class(
            observation_dim=observation_dim,
            action_dim=int(np.prod(self.action_space.shape)),
            config=intrinsic_config,
        )
        self.intrinsic_rewards_memory: list[float] = []
        self.intrinsic_info_memory: list[dict[str, float | bool]] = []

    def step(self, action):
        previous_time_index = self._time_index
        previous_observation = flatten_portfolio_observation(
            self._as_float32(self._state)
        )
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

        next_observation, external_reward, terminated, truncated, base_info = (
            super().step(action)
        )
        info = dict(base_info)
        time_advanced = self._time_index > previous_time_index

        if self.intrinsic_mode == "train" and time_advanced:
            combined_reward, intrinsic_info = (
                self.intrinsic_controller.process_transition(
                    observation=previous_observation,
                    action=policy_action,
                    next_observation=flatten_portfolio_observation(next_observation),
                    external_reward=float(external_reward),
                    portfolio_values=self._asset_memory["final"],
                )
            )
            self.intrinsic_rewards_memory.append(
                float(intrinsic_info["reward_intrinsic"])
            )
            self.intrinsic_info_memory.append(intrinsic_info.copy())
            info.update(intrinsic_info)
            return (
                next_observation,
                combined_reward,
                terminated,
                truncated,
                info,
            )

        info.update(self.intrinsic_controller.inactive_info(float(external_reward)))
        return next_observation, float(external_reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.intrinsic_rewards_memory = []
        self.intrinsic_info_memory = []
        if self.intrinsic_mode == "train":
            self.intrinsic_controller.observe_initial(
                flatten_portfolio_observation(observation)
            )
        return observation, info

    def set_intrinsic_mode(self, mode: IntrinsicMode) -> None:
        self._validate_intrinsic_mode(mode)
        self.intrinsic_mode = mode

    def save_intrinsic_state(self, path: str | Path) -> None:
        self.intrinsic_controller.save(path)

    def load_intrinsic_state(self, path: str | Path) -> None:
        self.intrinsic_controller.load(path)

    @staticmethod
    def _validate_intrinsic_mode(mode: str) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError("intrinsic_mode must be either 'train' or 'eval'")
