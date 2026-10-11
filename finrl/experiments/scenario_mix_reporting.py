"""Align scenario-mix episode rewards and checkpoints on actual PPO timesteps.

``timesteps`` remains the nominal checkpoint ID used by existing filenames and
joins. ``actual_timesteps`` is the environment-transition count used for plots
and episode buckets. Legacy logs used one env, 2,048-step PPO rollouts, and a
separate ``learn(eval_freq, reset_num_timesteps=False)`` call per checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _positive_steps(values, name):
    steps = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(steps) & (steps > 0) & (steps == np.floor(steps))):
        raise ValueError(f"{name} must contain positive integer timesteps")
    return steps.astype(np.int64)


def load_checkpoint_history(run: Path) -> pd.DataFrame:
    """Read validation logs, recovering legacy rollout boundaries when needed."""
    run = Path(run)
    history = pd.read_csv(run / "valid.csv")
    nominal = _positive_steps(history.timesteps, "timesteps")
    if not len(nominal) or np.any(np.diff(nominal) <= 0):
        raise ValueError(
            "checkpoint timesteps must be nonempty and strictly increasing"
        )
    if "actual_timesteps" not in history:
        config = json.loads((run / "config.json").read_text())
        frequency = _positive_steps([config["eval_freq"]], "eval_freq")[0]
        rollout = _positive_steps([config.get("rollout_steps", 2048)], "rollout_steps")[
            0
        ]
        if not np.array_equal(nominal, np.arange(1, len(nominal) + 1) * frequency):
            raise ValueError("legacy checkpoint schedule is incomplete or irregular")
        actual_interval = ((frequency + rollout - 1) // rollout) * rollout
        history["actual_timesteps"] = np.arange(1, len(nominal) + 1) * actual_interval
    actual = _positive_steps(history.actual_timesteps, "actual_timesteps")
    if np.any(np.diff(actual) <= 0) or np.any(actual < nominal):
        raise ValueError("actual_timesteps must increase and be >= nominal timesteps")
    history["actual_timesteps"] = actual
    return history


def load_training_history(run: Path) -> pd.DataFrame:
    """Average completed episodes over (previous checkpoint, checkpoint] intervals.

    Episodes spanning a checkpoint belong to the interval where they finish;
    episode rewards are not split or attributed to a single frozen policy.
    """
    history = load_checkpoint_history(run)
    episodes = pd.read_csv(Path(run) / "episodes.csv")
    ends = _positive_steps(episodes.timesteps, "episode timesteps")
    boundaries = history.actual_timesteps.to_numpy()
    bucket = np.searchsorted(boundaries, ends, side="left")
    if np.any(bucket == len(boundaries)):
        raise ValueError("episode log extends beyond the last checkpoint")
    if not episodes.source.isin(["real", "synthetic"]).all():
        raise ValueError("episode source must be real or synthetic")
    episodes = episodes.assign(checkpoint=boundaries[bucket])
    for source, column in (("real", "train_real"), ("synthetic", "train_syn")):
        rewards = (
            episodes[episodes.source == source].groupby("checkpoint").reward.mean()
        )
        history[column] = history.actual_timesteps.map(rewards)
    return history


def align_checkpoint_evaluations(evaluation, history):
    """Attach actual steps to test metrics without changing checkpoint join keys."""
    keys = ["group", "seed", "timesteps"]
    steps = history[keys + ["actual_timesteps"]].rename(
        columns={"actual_timesteps": "_history_steps"}
    )
    aligned = evaluation.merge(steps, on=keys, how="left", validate="many_to_one")
    if aligned._history_steps.isna().any():
        raise ValueError("evaluation contains a checkpoint missing from validation")
    if "actual_timesteps" in aligned and not np.array_equal(
        aligned.actual_timesteps, aligned._history_steps
    ):
        raise ValueError("evaluation and validation disagree on actual_timesteps")
    aligned["actual_timesteps"] = aligned.pop("_history_steps").astype(np.int64)
    return aligned
