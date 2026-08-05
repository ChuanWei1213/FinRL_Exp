"""Train PPO with optional surprise and deja vu intrinsic rewards."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: E402
from finrl.config import INDICATORS  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading_intrinsic import (
    IntrinsicRewardStockTradingEnv,
)  # noqa: E402
from finrl.meta.rewards import IntrinsicRewardConfig  # noqa: E402


VARIANT_WEIGHTS = {
    "surprise": (0.05, 0.0),
    "dejavu": (0.0, 0.05),
    "combined": (0.05, 0.05),
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

INTRINSIC_INFO_KEYS = (
    "reward_ext",
    "reward_intrinsic",
    "reward_surprise_raw",
    "reward_surprise",
    "reward_dejavu_raw",
    "reward_dejavu",
    "intrinsic_eta",
    "intrinsic_time_decay",
    "intrinsic_volatility_factor",
    "intrinsic_surprise_loss",
    "intrinsic_dejavu_loss",
)


class IntrinsicRewardLoggingCallback(BaseCallback):
    """Write intrinsic reward components from environment info to SB3 logs."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for key in INTRINSIC_INFO_KEYS:
            values = [float(info[key]) for info in infos if key in info]
            if values:
                self.logger.record(f"intrinsic/{key}", float(np.mean(values)))
        warmup_values = [
            float(info["intrinsic_warmup"])
            for info in infos
            if "intrinsic_warmup" in info
        ]
        if warmup_values:
            self.logger.record("intrinsic/warmup", float(np.mean(warmup_values)))
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("baseline", "surprise", "dejavu", "combined"),
        default="combined",
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=Path("train_data.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/intrinsic_reward")
    )
    return parser.parse_args()


def load_training_data(path: Path) -> pd.DataFrame:
    train = pd.read_csv(path)
    train = train.set_index(train.columns[0])
    train.index.names = [""]
    return train


def build_environment(
    train: pd.DataFrame,
    variant: str,
    total_timesteps: int,
    seed: int,
):
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }
    if variant == "baseline":
        return StockTradingEnv(df=train, **env_kwargs), None

    alpha, beta = VARIANT_WEIGHTS[variant]
    intrinsic_config = IntrinsicRewardConfig(
        total_timesteps=total_timesteps,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
    environment = IntrinsicRewardStockTradingEnv(
        df=train,
        intrinsic_config=intrinsic_config,
        intrinsic_mode="train",
        **env_kwargs,
    )
    return environment, intrinsic_config


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")

    train = load_training_data(args.data_path)
    environment, intrinsic_config = build_environment(
        train,
        variant=args.variant,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
    )
    vector_environment, _ = environment.get_sb_env()

    run_dir = args.output_dir / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(run_dir / "logs"), ["stdout", "csv", "tensorboard"])

    agent = DRLAgent(env=vector_environment)
    model = agent.get_model(
        "ppo",
        model_kwargs=PPO_PARAMS.copy(),
        seed=args.seed,
    )
    model.set_logger(logger)
    trained_model = agent.train_model(
        model=model,
        tb_log_name=f"ppo_{args.variant}",
        total_timesteps=args.total_timesteps,
        callbacks=[IntrinsicRewardLoggingCallback()],
    )
    trained_model.save(str(run_dir / "agent_ppo"))

    experiment_config = {
        "variant": args.variant,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "data_path": str(args.data_path),
        "ppo": PPO_PARAMS,
        "intrinsic_reward": (
            asdict(intrinsic_config) if intrinsic_config is not None else None
        ),
    }
    with (run_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_config, file, indent=2, sort_keys=True)

    if isinstance(environment, IntrinsicRewardStockTradingEnv):
        environment.save_intrinsic_state(run_dir / "intrinsic_reward.pt")


if __name__ == "__main__":
    main()
