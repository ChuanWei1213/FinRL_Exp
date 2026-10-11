"""Shared scenario-mix ticker order, PPO construction, and validation scores.

Keep these protocol choices independent of CLI entry points. Entry-specific date
ranges, logging, checkpoint retention, and evaluation schedules remain separate.
"""

from __future__ import annotations

import numpy as np

TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]


def make_ppo(env, seed, device, log_std_init=-2.0, ent_coef=0.0, tensorboard=None):
    from stable_baselines3 import PPO
    from finrl.experiments.notebook_utils import EIIEExtractor

    return PPO(
        "MultiInputPolicy",
        env,
        seed=seed,
        device=device,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        tensorboard_log=str(tensorboard) if tensorboard else None,
        policy_kwargs={
            "log_std_init": log_std_init,
            "features_extractor_class": EIIEExtractor,
            "features_extractor_kwargs": {"k_size": 3, "conv_mid": 2, "conv_final": 20},
        },
    )


def validation_score(daily, metric):
    v = daily.value.to_numpy()
    if metric == "final_value":
        return float(v[-1])
    logr = np.diff(np.log(v))
    if metric == "sharpe":
        return (
            float(logr.mean() / logr.std(ddof=1) * np.sqrt(252))
            if logr.std() > 0
            else -np.inf
        )
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    ann = float(np.exp(logr.mean() * 252) - 1)
    return ann / abs(mdd) if mdd < 0 else np.inf
