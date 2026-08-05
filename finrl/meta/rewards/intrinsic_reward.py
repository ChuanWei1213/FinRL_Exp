from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal

from finrl.meta.rewards.novelty_models import DejavuModel
from finrl.meta.rewards.novelty_models import SimpleReplayPool
from finrl.meta.rewards.novelty_models import SurpriseModel


@dataclass(frozen=True)
class IntrinsicRewardConfig:
    """Configuration for the surprise and deja vu intrinsic reward."""

    total_timesteps: int
    alpha: float = 0.05
    beta: float = 0.05
    kappa: float = 0.5
    warmup_steps: int = 1024
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_rate: float = 1e-3
    update_every: int = 1
    gradient_steps: int = 1
    latent_dim: int = 32
    bonus_clip: float = 5.0
    volatility_window: int = 20
    volatility_scale: float = 100.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.update_every <= 0 or self.gradient_steps <= 0:
            raise ValueError("update_every and gradient_steps must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.bonus_clip <= 0:
            raise ValueError("bonus_clip must be positive")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.volatility_scale < 0:
            raise ValueError("volatility_scale must be non-negative")


class RunningMeanStd:
    """Numerically stable running mean and variance for observations."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == self.mean.ndim:
            batch = batch.reshape((1,) + self.mean.shape)
        if batch.shape[1:] != self.mean.shape:
            raise ValueError(
                f"running-stat shape mismatch: expected {self.mean.shape}, got {batch.shape[1:]}"
            )
        if not np.all(np.isfinite(batch)):
            raise ValueError("running-stat update contains non-finite values")

        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        old_sum = self.var * self.count
        batch_sum = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = (old_sum + batch_sum + correction) / total_count
        self.count = float(total_count)

    def normalize(self, values: np.ndarray, clip: float = 10.0) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        normalized = (array - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("running-stat checkpoint shape does not match controller")
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = float(state["count"])


class RunningRMS:
    """Running root-mean-square scale used to stabilize positive bonuses."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean_square = 1.0
        self.count = float(epsilon)

    def update(self, value: float) -> None:
        finite_value = float(value)
        if not math.isfinite(finite_value):
            raise ValueError("bonus scale update contains a non-finite value")
        squared = max(finite_value, 0.0) ** 2
        total_count = self.count + 1.0
        self.mean_square += (squared - self.mean_square) / total_count
        self.count = total_count

    def scale(self, value: float, clip: float) -> float:
        positive_value = max(float(value), 0.0)
        scaled = positive_value / math.sqrt(self.mean_square + 1e-8)
        return float(np.clip(scaled, 0.0, clip))

    def state_dict(self) -> dict[str, float]:
        return {"mean_square": self.mean_square, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean_square = float(state["mean_square"])
        self.count = float(state["count"])


class StableSurpriseModel(SurpriseModel):
    """Stable Gaussian forward model built on the supplied SurpriseModel."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        std_min: float = 0.05,
        std_max: float = 5.0,
    ):
        if not 0 < std_min < std_max:
            raise ValueError("std bounds must satisfy 0 < std_min < std_max")
        super().__init__(observation_dim, action_dim, device)
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _distribution(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Normal:
        inputs = torch.cat([observations, actions], dim=-1)
        features = self.predictor(inputs)
        mean = self.mean_head(features)
        std = torch.clamp(
            self.std_head(features) + self.std_min,
            min=self.std_min,
            max=self.std_max,
        )
        return Normal(mean, std)

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
            negative_log_likelihood = -distribution.log_prob(next_observations).mean()
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
        loss = -distribution.log_prob(next_obs_tensor).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("surprise-model loss is not finite")
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


class StableDejavuModel(DejavuModel):
    """Autoencoder novelty model with dimension-independent MSE bonuses."""

    def __init__(
        self,
        observation_dim: int,
        device: torch.device,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
    ):
        super().__init__(observation_dim, device, latent_dim)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def compute_bonus(self, obs: np.ndarray) -> float:
        observations = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        with torch.no_grad():
            reconstruction = self.decoder(self.encoder(observations))
            error = torch.mean(torch.square(reconstruction - observations))
        return float(error.item())

    def train_batch(self, observations: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        reconstruction = self.decoder(self.encoder(obs_tensor))
        loss = torch.mean(torch.square(reconstruction - obs_tensor))
        if not torch.isfinite(loss):
            raise FloatingPointError("deja-vu-model loss is not finite")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()
        return float(loss.item())


class SeededReplayPool(SimpleReplayPool):
    """The supplied replay pool with deterministic, validated sampling."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_size: int = 10_000,
        seed: int = 0,
    ):
        super().__init__(observation_dim, action_dim, max_size)
        self._rng = np.random.default_rng(seed)

    def random_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay pool")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self._observations[indices].copy(),
            "actions": self._actions[indices].copy(),
            "next_observations": self._next_observations[indices].copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "size": self._size,
            "pointer": self._pointer,
            "observations": self._observations.copy(),
            "actions": self._actions.copy(),
            "next_observations": self._next_observations.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["max_size"]) != self.max_size:
            raise ValueError("replay checkpoint capacity does not match controller")
        observations = np.asarray(state["observations"], dtype=np.float32)
        actions = np.asarray(state["actions"], dtype=np.float32)
        next_observations = np.asarray(state["next_observations"], dtype=np.float32)
        if observations.shape != self._observations.shape:
            raise ValueError("replay observation shape does not match controller")
        if actions.shape != self._actions.shape:
            raise ValueError("replay action shape does not match controller")
        if next_observations.shape != self._next_observations.shape:
            raise ValueError("replay next-observation shape does not match controller")
        self._observations[...] = observations
        self._actions[...] = actions
        self._next_observations[...] = next_observations
        self._size = int(state["size"])
        self._pointer = int(state["pointer"])
        if not 0 <= self._size <= self.max_size:
            raise ValueError("invalid replay size in checkpoint")
        if not 0 <= self._pointer < self.max_size:
            raise ValueError("invalid replay pointer in checkpoint")
        self._rng.bit_generator.state = state["rng_state"]


class IntrinsicRewardController:
    """Compute, normalize, train, and checkpoint intrinsic rewards."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: IntrinsicRewardConfig,
    ):
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = self._resolve_device(config.device)

        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.observation_stats = RunningMeanStd((self.observation_dim,))
        self.surprise_scale = RunningRMS()
        self.dejavu_scale = RunningRMS()
        self.replay_pool = SeededReplayPool(
            self.observation_dim,
            self.action_dim,
            max_size=config.replay_capacity,
            seed=config.seed,
        )

        self.surprise_model = None
        if config.alpha > 0:
            self.surprise_model = StableSurpriseModel(
                self.observation_dim,
                self.action_dim,
                self.device,
                learning_rate=config.learning_rate,
            )

        self.dejavu_model = None
        if config.beta > 0:
            effective_latent_dim = min(
                config.latent_dim, max(1, self.observation_dim // 2)
            )
            self.dejavu_model = StableDejavuModel(
                self.observation_dim,
                self.device,
                latent_dim=effective_latent_dim,
                learning_rate=config.learning_rate,
            )

        self.global_step = 0
        self.last_surprise_loss = 0.0
        self.last_dejavu_loss = 0.0

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return device

    def observe_initial(self, observation: np.ndarray) -> None:
        initial = self._validate_vector(
            observation, self.observation_dim, "initial observation"
        )
        self.observation_stats.update(initial)

    @staticmethod
    def portfolio_volatility_factor(
        portfolio_values: list[float] | np.ndarray,
        window: int,
        scale: float,
    ) -> float:
        values = np.asarray(portfolio_values, dtype=np.float64).reshape(-1)
        if values.size < 3:
            return 0.0
        previous = values[:-1]
        valid = np.isfinite(previous) & np.isfinite(values[1:]) & (previous != 0)
        returns = values[1:][valid] / previous[valid] - 1.0
        if returns.size < 2:
            return 0.0
        sigma = float(np.std(returns[-window:], ddof=0))
        scaled_sigma = max(0.0, scale * sigma)
        return float(scaled_sigma / (1.0 + scaled_sigma))

    def time_decay(self) -> float:
        progress = min(self.global_step, self.config.total_timesteps)
        return float(
            math.exp(-self.config.kappa * progress / self.config.total_timesteps)
        )

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
        dejavu_raw = 0.0
        if self.dejavu_model is not None:
            dejavu_raw = self.dejavu_model.compute_bonus(normalized_obs)

        surprise_bonus = self.surprise_scale.scale(surprise_raw, self.config.bonus_clip)
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
            self.surprise_scale.update(surprise_raw)
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
        }
        return float(combined_reward), info

    def inactive_info(self, external_reward: float) -> dict[str, float | bool]:
        reward_ext = float(external_reward)
        return {
            "reward_ext": reward_ext,
            "reward_intrinsic": 0.0,
            "reward_total": reward_ext,
            "reward_surprise_raw": 0.0,
            "reward_surprise": 0.0,
            "reward_dejavu_raw": 0.0,
            "reward_dejavu": 0.0,
            "intrinsic_eta": 0.0,
            "intrinsic_time_decay": 0.0,
            "intrinsic_volatility_factor": 0.0,
            "intrinsic_surprise_loss": float(self.last_surprise_loss),
            "intrinsic_dejavu_loss": float(self.last_dejavu_loss),
            "intrinsic_warmup": bool(self.global_step < self.config.warmup_steps),
        }

    def _maybe_train_models(self) -> None:
        if len(self.replay_pool) < self.config.batch_size:
            return
        if self.global_step % self.config.update_every != 0:
            return
        for _ in range(self.config.gradient_steps):
            batch = self.replay_pool.random_batch(self.config.batch_size)
            observations = self.observation_stats.normalize(batch["observations"])
            next_observations = self.observation_stats.normalize(
                batch["next_observations"]
            )
            if self.surprise_model is not None:
                self.last_surprise_loss = self.surprise_model.train_batch(
                    observations, batch["actions"], next_observations
                )
            if self.dejavu_model is not None:
                self.last_dejavu_loss = self.dejavu_model.train_batch(observations)

    @staticmethod
    def _validate_vector(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (expected_dim,):
            raise ValueError(
                f"{name} must have shape ({expected_dim},), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array.copy()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": self.CHECKPOINT_VERSION,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "global_step": self.global_step,
            "last_surprise_loss": self.last_surprise_loss,
            "last_dejavu_loss": self.last_dejavu_loss,
            "observation_stats": self.observation_stats.state_dict(),
            "surprise_scale": self.surprise_scale.state_dict(),
            "dejavu_scale": self.dejavu_scale.state_dict(),
            "replay_pool": self.replay_pool.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.surprise_model is not None:
            state["surprise_model"] = {
                "predictor": self.surprise_model.predictor.state_dict(),
                "mean_head": self.surprise_model.mean_head.state_dict(),
                "std_head": self.surprise_model.std_head.state_dict(),
                "optimizer": self.surprise_model.optimizer.state_dict(),
            }
        if self.dejavu_model is not None:
            state["dejavu_model"] = {
                "encoder": self.dejavu_model.encoder.state_dict(),
                "decoder": self.dejavu_model.decoder.state_dict(),
                "optimizer": self.dejavu_model.optimizer.state_dict(),
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["version"]) != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported intrinsic-reward checkpoint version")
        if int(state["observation_dim"]) != self.observation_dim:
            raise ValueError("checkpoint observation dimension does not match")
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint action dimension does not match")
        checkpoint_config = state["config"]
        if bool(checkpoint_config["alpha"] > 0) != bool(
            self.surprise_model is not None
        ):
            raise ValueError("checkpoint surprise-model configuration does not match")
        if bool(checkpoint_config["beta"] > 0) != bool(self.dejavu_model is not None):
            raise ValueError("checkpoint deja-vu-model configuration does not match")

        self.global_step = int(state["global_step"])
        self.last_surprise_loss = float(state["last_surprise_loss"])
        self.last_dejavu_loss = float(state["last_dejavu_loss"])
        self.observation_stats.load_state_dict(state["observation_stats"])
        self.surprise_scale.load_state_dict(state["surprise_scale"])
        self.dejavu_scale.load_state_dict(state["dejavu_scale"])
        self.replay_pool.load_state_dict(state["replay_pool"])
        torch.set_rng_state(state["torch_rng_state"].cpu())

        if self.surprise_model is not None:
            surprise_state = state["surprise_model"]
            self.surprise_model.predictor.load_state_dict(surprise_state["predictor"])
            self.surprise_model.mean_head.load_state_dict(surprise_state["mean_head"])
            self.surprise_model.std_head.load_state_dict(surprise_state["std_head"])
            self.surprise_model.optimizer.load_state_dict(surprise_state["optimizer"])
        if self.dejavu_model is not None:
            dejavu_state = state["dejavu_model"]
            self.dejavu_model.encoder.load_state_dict(dejavu_state["encoder"])
            self.dejavu_model.decoder.load_state_dict(dejavu_state["decoder"])
            self.dejavu_model.optimizer.load_state_dict(dejavu_state["optimizer"])

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, path: str | Path) -> None:
        try:
            state = torch.load(Path(path), map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(Path(path), map_location=self.device)
        self.load_state_dict(state)
