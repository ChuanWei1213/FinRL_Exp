from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from finrl.meta.rewards.intrinsic_reward import IntrinsicRewardConfig
from finrl.meta.rewards.intrinsic_reward import IntrinsicRewardController
from finrl.meta.rewards.intrinsic_reward import RunningMeanStd
from finrl.meta.rewards.intrinsic_reward import StableSurpriseModel


class PaperFaithfulSurpriseModel(StableSurpriseModel):
    """Diagonal-Gaussian NLL summed over dimensions, as in the paper formula."""

    def compute_bonus(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        observations = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        actions = torch.as_tensor(
            np.asarray(action, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        next_observations = torch.as_tensor(
            np.asarray(next_obs, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        with torch.no_grad():
            distribution = self._distribution(observations, actions)
            negative_log_likelihood = -distribution.log_prob(next_observations).sum(
                dim=-1
            )
        return float(negative_log_likelihood.item())

    def train_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
    ) -> float:
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        action_tensor = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        next_obs_tensor = torch.as_tensor(
            next_observations, dtype=torch.float32, device=self.device
        )
        distribution = self._distribution(obs_tensor, action_tensor)
        loss = -distribution.log_prob(next_obs_tensor).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("paper-faithful surprise loss is not finite")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.predictor.parameters())
            + list(self.mean_head.parameters())
            + list(self.std_head.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()
        return float(loss.item())


class _VariantSurpriseController(IntrinsicRewardController):
    """Shared transition flow for alternative Surprise score transformations."""

    CONTROLLER_KIND = "variant"

    def _transform_surprise(
        self, surprise_raw: float
    ) -> tuple[float, dict[str, float]]:
        raise NotImplementedError

    def _update_surprise_statistics(self, surprise_raw: float) -> None:
        raise NotImplementedError

    def process_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        external_reward: float,
        portfolio_values: list[float] | np.ndarray,
    ) -> tuple[float, dict[str, float | bool]]:
        obs = self._validate_vector(observation, self.observation_dim, "observation")
        act = self._validate_vector(action, self.action_dim, "action")
        next_obs = self._validate_vector(
            next_observation, self.observation_dim, "next observation"
        )
        reward_ext = float(external_reward)
        if not math.isfinite(reward_ext):
            raise ValueError("external reward must be finite")

        normalized_obs = self.observation_stats.normalize(obs)
        normalized_next_obs = self.observation_stats.normalize(next_obs)

        surprise_raw = 0.0
        if self.surprise_model is not None:
            surprise_raw = self.surprise_model.compute_bonus(
                normalized_obs, act, normalized_next_obs
            )
        surprise_bonus, surprise_diagnostics = self._transform_surprise(surprise_raw)

        dejavu_raw = 0.0
        if self.dejavu_model is not None:
            dejavu_raw = self.dejavu_model.compute_bonus(normalized_obs)
        dejavu_bonus = self.dejavu_scale.scale(dejavu_raw, self.config.bonus_clip)

        time_decay = self.time_decay()
        volatility_factor = self.portfolio_volatility_factor(
            portfolio_values,
            self.config.volatility_window,
            self.config.volatility_scale,
        )
        eta = time_decay * volatility_factor
        in_warmup = self.global_step < self.config.warmup_steps
        intrinsic_reward = 0.0
        if not in_warmup:
            intrinsic_reward = eta * (
                self.config.alpha * surprise_bonus + self.config.beta * dejavu_bonus
            )
        combined_reward = reward_ext + intrinsic_reward

        self.replay_pool.add_sample(obs, act, next_obs)
        self.observation_stats.update(next_obs)
        if self.surprise_model is not None:
            self._update_surprise_statistics(surprise_raw)
        if self.dejavu_model is not None:
            self.dejavu_scale.update(dejavu_raw)
        self.global_step += 1
        self._maybe_train_models()

        info: dict[str, float | bool] = {
            "reward_ext": reward_ext,
            "reward_intrinsic": float(intrinsic_reward),
            "reward_total": float(combined_reward),
            "reward_surprise_raw": float(surprise_raw),
            "reward_surprise": float(surprise_bonus),
            "reward_dejavu_raw": float(dejavu_raw),
            "reward_dejavu": float(dejavu_bonus),
            "intrinsic_eta": float(eta),
            "intrinsic_time_decay": float(time_decay),
            "intrinsic_volatility_factor": float(volatility_factor),
            "intrinsic_surprise_loss": float(self.last_surprise_loss),
            "intrinsic_dejavu_loss": float(self.last_dejavu_loss),
            "intrinsic_warmup": bool(in_warmup),
            **surprise_diagnostics,
        }
        return float(combined_reward), info

    def inactive_info(self, external_reward: float) -> dict[str, float | bool]:
        return {
            **super().inactive_info(external_reward),
            "intrinsic_surprise_center": 0.0,
            "intrinsic_surprise_std": 0.0,
            "intrinsic_surprise_z": 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["surprise_controller_kind"] = self.CONTROLLER_KIND
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("surprise_controller_kind") != self.CONTROLLER_KIND:
            raise ValueError("surprise-controller checkpoint kind does not match")
        super().load_state_dict(state)


class PaperFaithfulIntrinsicRewardController(_VariantSurpriseController):
    """Use summed Gaussian NLL directly, without ReLU, RMS scaling, or clipping."""

    CONTROLLER_KIND = "paper_faithful"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: IntrinsicRewardConfig,
    ):
        super().__init__(observation_dim, action_dim, config)
        if config.alpha > 0:
            self.surprise_model = PaperFaithfulSurpriseModel(
                observation_dim,
                action_dim,
                self.device,
                learning_rate=config.learning_rate,
            )

    def _transform_surprise(
        self, surprise_raw: float
    ) -> tuple[float, dict[str, float]]:
        return float(surprise_raw), {
            "intrinsic_surprise_center": 0.0,
            "intrinsic_surprise_std": 0.0,
            "intrinsic_surprise_z": 0.0,
        }

    def _update_surprise_statistics(self, surprise_raw: float) -> None:
        return None


class RobustIntrinsicRewardController(_VariantSurpriseController):
    """Use positive running-z Surprise based on dimension-mean Gaussian NLL."""

    CONTROLLER_KIND = "robust_centered_z"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: IntrinsicRewardConfig,
    ):
        super().__init__(observation_dim, action_dim, config)
        self.surprise_reference = RunningMeanStd(())

    def _transform_surprise(
        self, surprise_raw: float
    ) -> tuple[float, dict[str, float]]:
        center = float(self.surprise_reference.mean)
        standard_deviation = float(np.sqrt(self.surprise_reference.var + 1e-8))
        z_score = (float(surprise_raw) - center) / standard_deviation
        bonus = float(np.clip(max(z_score, 0.0), 0.0, self.config.bonus_clip))
        return bonus, {
            "intrinsic_surprise_center": center,
            "intrinsic_surprise_std": standard_deviation,
            "intrinsic_surprise_z": float(z_score),
        }

    def _update_surprise_statistics(self, surprise_raw: float) -> None:
        self.surprise_reference.update(np.asarray(surprise_raw, dtype=np.float64))

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["surprise_reference"] = self.surprise_reference.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.surprise_reference.load_state_dict(state["surprise_reference"])
