from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.rewards.intrinsic_reward import IntrinsicRewardConfig
from finrl.meta.rewards.intrinsic_reward import IntrinsicRewardController


IntrinsicMode = Literal["train", "eval"]


class IntrinsicRewardStockTradingEnv(StockTradingEnv):
    """StockTradingEnv with an opt-in, training-only intrinsic reward."""

    def __init__(
        self,
        *args,
        intrinsic_config: IntrinsicRewardConfig,
        intrinsic_mode: IntrinsicMode = "train",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._validate_intrinsic_mode(intrinsic_mode)
        self.intrinsic_mode: IntrinsicMode = intrinsic_mode
        self.intrinsic_controller = IntrinsicRewardController(
            observation_dim=int(np.prod(self.observation_space.shape)),
            action_dim=int(np.prod(self.action_space.shape)),
            config=intrinsic_config,
        )
        self.intrinsic_rewards_memory: list[float] = []
        self.intrinsic_info_memory: list[dict[str, float | bool]] = []
        if self.intrinsic_mode == "train":
            self.intrinsic_controller.observe_initial(
                np.asarray(self.state, dtype=np.float32)
            )

    def step(self, actions):
        previous_day = self.day
        previous_state = np.asarray(self.state, dtype=np.float32).copy()
        policy_action = np.asarray(actions, dtype=np.float32).reshape(-1).copy()

        next_state, external_reward, terminated, truncated, base_info = super().step(
            actions
        )
        info = dict(base_info)
        day_advanced = self.day > previous_day

        if self.intrinsic_mode == "train" and day_advanced:
            combined_reward, intrinsic_info = (
                self.intrinsic_controller.process_transition(
                    observation=previous_state,
                    action=policy_action,
                    next_observation=np.asarray(next_state, dtype=np.float32),
                    external_reward=float(external_reward),
                    portfolio_values=self.asset_memory,
                )
            )
            self.intrinsic_rewards_memory.append(
                float(intrinsic_info["reward_intrinsic"])
            )
            self.intrinsic_info_memory.append(intrinsic_info.copy())
            info.update(intrinsic_info)
            return next_state, combined_reward, terminated, truncated, info

        info.update(self.intrinsic_controller.inactive_info(float(external_reward)))
        return next_state, float(external_reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.intrinsic_rewards_memory = []
        self.intrinsic_info_memory = []
        if self.intrinsic_mode == "train":
            self.intrinsic_controller.observe_initial(
                np.asarray(observation, dtype=np.float32)
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
