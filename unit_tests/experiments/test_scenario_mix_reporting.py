from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from finrl.experiments.scenario_mix_reporting import (
    align_checkpoint_evaluations,
    load_checkpoint_history,
    load_training_history,
)


def write_run(tmp_path, history, episodes, config=None):
    history.to_csv(tmp_path / "valid.csv", index=False)
    episodes.to_csv(tmp_path / "episodes.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(config or {"eval_freq": 25000}))
    return tmp_path


def test_legacy_buckets_include_rollout_overshoot_and_final_tail(tmp_path):
    history = pd.DataFrame({"timesteps": np.arange(25000, 500001, 25000)})
    episodes = pd.DataFrame(
        {
            "timesteps": [25000, 25001, 26624, 26625, 53248, 500001, 532480],
            "reward": [1, 3, 5, 7, 9, 11, 13],
            "source": ["real", "real", "real", "real", "synthetic", "real", "real"],
        }
    )
    result = load_training_history(write_run(tmp_path, history, episodes))
    assert result.actual_timesteps.iloc[0] == 26624
    assert result.actual_timesteps.iloc[-1] == 532480
    assert result.train_real.iloc[0] == 3  # includes 25,001 .. 26,624
    assert result.train_real.iloc[1] == 7  # boundary+1 starts the next bucket
    assert result.train_syn.iloc[1] == 9
    assert result.train_real.iloc[18] == 11
    assert result.train_real.iloc[19] == 13  # used to be discarded
    assert result.timesteps.equals(history.timesteps)


def test_explicit_actual_steps_override_legacy_defaults(tmp_path):
    history = pd.DataFrame({"timesteps": [9, 18], "actual_timesteps": [16, 32]})
    episodes = pd.DataFrame(
        {"timesteps": [16, 17, 32], "reward": [2, 4, 8], "source": ["real"] * 3}
    )
    result = load_training_history(write_run(tmp_path, history, episodes))
    np.testing.assert_array_equal(result.train_real, [2, 6])
    assert result.train_syn.isna().all()


def test_legacy_rollout_metadata_and_empty_episode_source(tmp_path):
    history = pd.DataFrame({"timesteps": [9, 18]})
    episodes = pd.DataFrame(columns=["timesteps", "reward", "source"])
    result = load_training_history(
        write_run(tmp_path, history, episodes, {"eval_freq": 9, "rollout_steps": 8})
    )
    np.testing.assert_array_equal(result.actual_timesteps, [16, 32])
    assert result[["train_real", "train_syn"]].isna().all().all()


@pytest.mark.parametrize("steps", [[25000, 75000], [25000, 25000], [25000, -50000]])
def test_invalid_legacy_schedule_is_not_guessed(tmp_path, steps):
    run = write_run(tmp_path, pd.DataFrame({"timesteps": steps}), pd.DataFrame())
    with pytest.raises(ValueError):
        load_checkpoint_history(run)


def test_episode_after_last_checkpoint_is_not_silently_dropped(tmp_path):
    run = write_run(
        tmp_path,
        pd.DataFrame({"timesteps": [25000]}),
        pd.DataFrame(
            {
                "timesteps": [26625],
                "reward": [1],
                "source": ["real"],
            }
        ),
    )
    with pytest.raises(ValueError, match="beyond the last checkpoint"):
        load_training_history(run)


def test_test_metrics_join_by_checkpoint_id_and_validate_actual_steps():
    history = pd.DataFrame(
        {
            "group": ["A", "B90"],
            "seed": [0, 0],
            "timesteps": [25000, 25000],
            "actual_timesteps": [26624, 26624],
        }
    )
    evaluation = pd.DataFrame(
        {
            "group": ["B90", "A", "A"],
            "seed": [0] * 3,
            "timesteps": [25000] * 3,
            "period": ["2020", "2020", "gfc"],
        }
    )
    aligned = align_checkpoint_evaluations(evaluation, history)
    np.testing.assert_array_equal(aligned.actual_timesteps, [26624] * 3)
    pd.testing.assert_frame_equal(aligned[evaluation.columns], evaluation)
    pd.testing.assert_frame_equal(
        align_checkpoint_evaluations(aligned, history), aligned
    )
    with pytest.raises(ValueError, match="disagree"):
        align_checkpoint_evaluations(aligned.assign(actual_timesteps=25000), history)
    with pytest.raises(ValueError, match="missing from validation"):
        align_checkpoint_evaluations(evaluation.assign(timesteps=50000), history)


def test_real_ppo_rollout_boundaries_align_with_completed_episodes(tmp_path):
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    episodes = []

    class EpisodeLog(BaseCallback):
        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    episodes.append(
                        {
                            "timesteps": self.num_timesteps,
                            "reward": info["episode"]["r"],
                            "source": "real",
                        }
                    )
            return True

    env = Monitor(gym.make("Pendulum-v1", max_episode_steps=5))
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=0,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
        )
        history = []
        for nominal in (9, 18):
            model.learn(9, reset_num_timesteps=False, callback=EpisodeLog())
            history.append(
                {"timesteps": nominal, "actual_timesteps": model.num_timesteps}
            )
        result = load_training_history(
            write_run(tmp_path, pd.DataFrame(history), pd.DataFrame(episodes))
        )
        np.testing.assert_array_equal(result.actual_timesteps, [16, 32])
        assert [e["timesteps"] for e in episodes] == [5, 10, 15, 20, 25, 30]
        assert result.train_real.iloc[0] == pytest.approx(
            np.mean([e["reward"] for e in episodes[:3]])
        )
        assert result.train_real.iloc[1] == pytest.approx(
            np.mean([e["reward"] for e in episodes[3:]])
        )
    finally:
        env.close()
