import pandas as pd
from pandas.testing import assert_frame_equal

from finrl.experiments.stock_intrinsic import build_training_environment
from finrl.experiments.stock_intrinsic import plan_variant_training
from finrl.experiments.stock_intrinsic import stock_training_cache_key
from finrl.experiments.stock_intrinsic import train_variant


def _market_frame() -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-02", periods=4)):
        for ticker, base_price in (("AAA", 10.0), ("BBB", 20.0)):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "tic": ticker,
                    "close": base_price + day,
                    "macd": 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.index = frame["date"].factorize()[0]
    return frame


def _environment_kwargs() -> dict:
    return {
        "stock_dim": 2,
        "hmax": 10,
        "initial_amount": 1_000,
        "num_stock_shares": [0, 0],
        "buy_cost_pct": [0.001, 0.001],
        "sell_cost_pct": [0.001, 0.001],
        "reward_scaling": 1e-4,
        "state_space": 7,
        "action_space": 2,
        "tech_indicator_list": ["macd"],
        "print_verbosity": 10_000,
    }


def _ppo_kwargs() -> dict:
    return {
        "n_steps": 4,
        "batch_size": 4,
        "n_epochs": 1,
        "device": "cpu",
    }


def test_training_cache_key_changes_with_semantic_inputs():
    frame = _market_frame()
    arguments = {
        "train": frame,
        "validation": frame,
        "variant": "baseline",
        "total_timesteps": 8,
        "eval_freq": 4,
        "seed": 0,
        "warmup_steps": 4,
        "env_kwargs": _environment_kwargs(),
        "ppo_kwargs": _ppo_kwargs(),
    }

    first_key, payload = stock_training_cache_key(**arguments)
    repeated_key, _ = stock_training_cache_key(**arguments)
    seed_key, _ = stock_training_cache_key(**{**arguments, "seed": 1})

    assert first_key == repeated_key
    assert first_key != seed_key
    assert payload["initial_allocation"] == "cash"


def test_full_intrinsic_run_caps_replay_and_uses_requested_device():
    frame = _market_frame()
    ppo_kwargs = _ppo_kwargs()
    _, payload = stock_training_cache_key(
        train=frame,
        validation=frame,
        variant="combined",
        total_timesteps=200_000,
        eval_freq=20_000,
        seed=0,
        warmup_steps=1_024,
        env_kwargs=_environment_kwargs(),
        ppo_kwargs=ppo_kwargs,
    )
    environment = build_training_environment(
        train=frame,
        variant="combined",
        total_timesteps=200_000,
        seed=0,
        warmup_steps=1_024,
        env_kwargs=_environment_kwargs(),
        device="cpu",
    )

    assert payload["intrinsic_reward"]["replay_capacity"] == 100_000
    assert payload["intrinsic_reward"]["device"] == "cpu"
    assert environment.intrinsic_controller.config.replay_capacity == 100_000
    assert environment.intrinsic_controller.device.type == "cpu"


def test_train_variant_reuses_complete_cache(tmp_path):
    frame = _market_frame()
    arguments = {
        "train": frame,
        "validation": frame,
        "variant": "baseline",
        "total_timesteps": 8,
        "eval_freq": 4,
        "seed": 0,
        "warmup_steps": 4,
        "env_kwargs": _environment_kwargs(),
        "ppo_kwargs": _ppo_kwargs(),
        "output_dir": tmp_path / "outputs",
        "cache_dir": tmp_path / "cache",
        "show_progress": False,
    }

    plan_arguments = {
        key: value
        for key, value in arguments.items()
        if key not in {"output_dir", "show_progress"}
    }
    initial_plan = plan_variant_training(**plan_arguments)
    trained = train_variant(**arguments, training_plan=initial_plan)
    cached_plan = plan_variant_training(**plan_arguments)
    cached = train_variant(**arguments, training_plan=cached_plan)

    assert initial_plan.cache_hit is False
    assert trained.cache_hit is False
    assert cached_plan.cache_hit is True
    assert cached.cache_hit is True
    assert cached.cache_key == trained.cache_key
    assert_frame_equal(cached.intrinsic_log, trained.intrinsic_log)
    assert {
        "train_reward_mean",
        "real_valid_reward",
        "real_valid_turnover",
        "real_valid_max_weight",
        "is_best",
    }.issubset(cached.validation_log.columns)
    assert (tmp_path / "cache" / trained.cache_key / "agent_ppo.zip").is_file()
