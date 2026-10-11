from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from finrl.experiments import evaluate_scenario_mix as evaluate
from finrl.experiments import evaluate_scenario_mix_checkpoints as checkpoints
from finrl.experiments import scenario_mix as S
from finrl.experiments import scenario_mix_common as common
from finrl.experiments import train_scenario_mix as train
from finrl.experiments import train_scenario_mix_logged as logged
from finrl.experiments.scenario_mix_reporting import load_training_history


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = [
    ("train_scenario_mix_ppo", train),
    ("log_training_rewards", logged),
    ("evaluate_scenario_mix", evaluate),
    ("evaluate_checkpoints", checkpoints),
]


def test_cli_defaults_preserve_distinct_protocols():
    data = ["--prices", "prices.csv", "--synthetic", "paths.npz", "--out", "out"]
    shared = {
        "prices": Path("prices.csv"),
        "synthetic": Path("paths.npz"),
        "out": Path("out"),
        "valid": ["2018-01-01", "2019-12-31"],
        "timesteps": 500_000,
        "eval_freq": 25_000,
        "commission": 0.0025,
        "k_max": 21,
        "log_std_init": -2.0,
        "ent_coef": 0.0,
    }
    assert vars(train.parse(data)) == {
        **shared,
        "train": ["2009-04-01", "2017-12-31"],
        "test": ["2020-01-02", "2025-12-31"],
        "groups": ["A", "B20", "B50"],
        "custom_group": [],
        "seeds": [0, 1, 2],
        "device": "cpu",
        "select_metric": "final_value",
        "tensorboard": None,
        "crash_diag_n": 100,
        "torch_threads": 1,
    }
    assert vars(
        logged.build_parser().parse_args(
            data + ["--group", "B90", "--p", "0.9", "--seed", "2"]
        )
    ) == {
        **shared,
        "train": ["2010-01-01", "2017-12-31"],
        "group": "B90",
        "p": 0.9,
        "seed": 2,
        "train_commission": None,
    }
    assert vars(
        evaluate.build_parser().parse_args(
            [
                "--runs",
                "runs",
                "--prices",
                "prices.csv",
                "--out",
                "out",
            ]
        )
    ) == {
        "runs": Path("runs"),
        "prices": Path("prices.csv"),
        "out": Path("out"),
        "test": ["2020-01-02", "2025-12-31"],
        "only": None,
        "device": "cpu",
        "torch_threads": 1,
    }
    assert vars(
        checkpoints.build_parser().parse_args(
            [
                "--root",
                "runs",
                "--prices",
                "prices.csv",
            ]
        )
    ) == {
        "root": Path("runs"),
        "prices": Path("prices.csv"),
        "test": ["2020-01-02", "2025-12-31"],
        "extra": None,
        "commission": 0.0025,
        "workers": 15,
    }
    assert common.TICKERS == [
        "AAPL",
        "AMGN",
        "CRM",
        "CSCO",
        "IBM",
        "INTC",
        "MSFT",
        "NKE",
        "VZ",
        "WMT",
    ]
    assert train.GROUPS == {"A": 0.0, "B20": 0.2, "B50": 0.5}


@pytest.mark.parametrize("legacy,module", ENTRY_POINTS)
def test_legacy_imports_and_both_cli_forms(legacy, module, tmp_path):
    # examples/ is a script directory; an installed package may share its name.
    spec = importlib.util.spec_from_file_location(
        legacy, PROJECT_ROOT / "examples" / f"{legacy}.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for name in (
        "main",
        "TICKERS",
        "make_ppo",
        "validation_score",
        "SourceInfo",
        "eval_run",
        "parse",
    ):
        if hasattr(module, name):
            assert getattr(old, name) is getattr(module, name)
    # Direct legacy scripts must still work outside the repository without PYTHONPATH.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    for args, cwd in [
        ([str(PROJECT_ROOT / "examples" / f"{legacy}.py")], tmp_path),
        (["-m", module.__name__], PROJECT_ROOT),
    ]:
        result = subprocess.run(
            [sys.executable, *args, "--help"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "--prices" in result.stdout
        assert "--help" in result.stdout


def test_shared_ppo_configuration_is_unchanged(monkeypatch):
    import stable_baselines3
    from finrl.experiments.notebook_utils import EIIEExtractor

    calls = []
    sentinel = object()

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(stable_baselines3, "PPO", capture)
    env = object()
    assert common.make_ppo(env, 2, "cpu") is sentinel
    expected = {
        "seed": 2,
        "device": "cpu",
        "verbose": 0,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "tensorboard_log": None,
        "policy_kwargs": {
            "log_std_init": -2.0,
            "features_extractor_class": EIIEExtractor,
            "features_extractor_kwargs": {"k_size": 3, "conv_mid": 2, "conv_final": 20},
        },
    }
    assert calls[0] == (("MultiInputPolicy", env), expected)
    common.make_ppo(env, 2, "cpu", -0.5, 0.01, Path("logs"))
    expected["ent_coef"] = 0.01
    expected["tensorboard_log"] = "logs"
    expected["policy_kwargs"]["log_std_init"] = -0.5
    assert calls[1] == (("MultiInputPolicy", env), expected)


def test_validation_scores_keep_numerics_and_degenerate_cases():
    daily = pd.DataFrame({"value": [1.0, 1.1, 0.99, 1.08]})
    returns = np.log([1.1, 0.9, 1.08 / 0.99])
    assert common.validation_score(daily, "final_value") == 1.08
    assert common.validation_score(daily, "sharpe") == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    )
    assert common.validation_score(daily, "calmar") == pytest.approx(
        (1.08 ** (252 / 3) - 1) / 0.1
    )
    flat = pd.DataFrame({"value": [1.0, 1.0, 1.0]})
    assert common.validation_score(flat, "sharpe") == -np.inf
    assert common.validation_score(flat, "calmar") == np.inf


@pytest.fixture
def market_files(tmp_path):
    dates = pd.bdate_range("2010-01-01", periods=180)
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.005, (180, 10)), axis=0))
    prices = tmp_path / "prices.csv"
    S.episode_frame(dates, close, close * 1.01, close * 0.99, common.TICKERS).to_csv(
        prices, index=False
    )
    anchors = [80, 90]
    paths = np.stack([close[a + 1 : a + 22] for a in anchors])
    raw = np.zeros((2, 21, 30))
    raw[:, 0, ::3] = np.log(paths[:, 0] / close[anchors])
    synthetic = tmp_path / "paths.npz"
    np.savez(
        synthetic,
        tickers=common.TICKERS,
        close=paths,
        high=paths * 1.01,
        low=paths * 0.99,
        raw_features=raw,
        scenario=[0, 1],
        anchor_dates=np.array([str(dates[a].date()) for a in anchors]),
    )
    date = lambda i: str(dates[i].date())
    return (
        prices,
        synthetic,
        [date(0), date(119)],
        [date(120), date(139)],
        [date(140), date(179)],
    )


@pytest.fixture
def small_ppo(monkeypatch):
    """Use real PPO and the same extractor, reducing only test training cost."""
    import stable_baselines3

    original_ppo = stable_baselines3.PPO
    models = []

    def factory(*args, **kwargs):
        with monkeypatch.context() as patch:

            def reduced(policy, env, **settings):
                settings.update(n_steps=16, batch_size=16, n_epochs=1)
                return original_ppo(policy, env, **settings)

            patch.setattr(stable_baselines3, "PPO", reduced)
            model = common.make_ppo(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(train, "make_ppo", factory)
    monkeypatch.setattr(logged, "make_ppo", factory)
    yield models
    for model in models:
        model.get_env().close()


def file_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_train_select_save_and_reevaluate_preserve_artifacts(
    market_files, small_ppo, monkeypatch, tmp_path
):
    from stable_baselines3 import PPO

    prices, synthetic, training, valid, test = market_files
    run = tmp_path / "trained" / "B50_seed0"
    args = [
        "--prices",
        str(prices),
        "--synthetic",
        str(synthetic),
        "--out",
        str(run),
        "--train",
        *training,
        "--valid",
        *valid,
        "--test",
        *test,
        "--groups",
        "B50",
        "--seeds",
        "0",
        "--timesteps",
        "50",
        "--eval-freq",
        "25",
        "--k-max",
        "10",
        "--crash-diag-n",
        "1",
    ]
    # A tied validation score must keep the first checkpoint, not the last.
    monkeypatch.setattr(train, "validation_score", lambda daily, metric: 1.0)
    train.main(args)
    history = pd.read_csv(run / "B50_seed0_valid.csv")
    assert history.timesteps.tolist() == [25, 50]
    assert history.actual_timesteps.tolist() == [32, 64]
    assert PPO.load(run / "B50_seed0.zip").num_timesteps == 32
    assert (run / "B50_seed0_crash_response.csv").exists()
    assert (run / "summary_by_group.csv").exists()
    assert set(pd.read_csv(run / "summary.csv").group) == {
        "B50",
        "buy_hold_1_11",
        "buy_hold_1_10",
    }
    before = file_hashes(run)
    with pytest.raises(FileExistsError):
        train.main(args)
    out = tmp_path / "reevaluated"
    eval_args = [
        "--runs",
        str(run.parent),
        "--prices",
        str(prices),
        "--test",
        *test,
        "--only",
        "B50_seed0",
        "--out",
        str(out),
    ]
    evaluate.main(eval_args)
    destination = out / run.name
    for filename in [
        "B50_seed0_test_daily.csv",
        "B50_seed0_valid.csv",
        "buy_hold_1_11_test_daily.csv",
        "buy_hold_1_10_test_daily.csv",
    ]:
        pd.testing.assert_frame_equal(
            pd.read_csv(run / filename), pd.read_csv(destination / filename)
        )
    assert file_hashes(run) == before
    with pytest.raises(FileExistsError):
        evaluate.main(eval_args)


@pytest.mark.parametrize("train_commission", [None, 0.0])
def test_logged_training_and_checkpoint_worker(
    market_files, small_ppo, tmp_path, train_commission
):
    prices, synthetic, training, valid, test = market_files
    root = tmp_path / "logged"
    run = root / "B50_seed0"
    args = [
        "--prices",
        str(prices),
        "--synthetic",
        str(synthetic),
        "--out",
        str(run),
        "--train",
        *training,
        "--valid",
        *valid,
        "--group",
        "B50",
        "--p",
        "0.5",
        "--seed",
        "0",
        "--timesteps",
        "50",
        "--eval-freq",
        "25",
        "--k-max",
        "10",
    ]
    if train_commission is not None:
        args += ["--train-commission", str(train_commission)]
    logged.main(args)
    assert small_ppo[0].get_env().envs[0].unwrapped.commission == (
        0.0025 if train_commission is None else train_commission
    )
    config = json.loads((run / "config.json").read_text())
    assert config["rollout_steps"] == 16
    assert config["commission"] == 0.0025
    assert config["train_commission"] == train_commission
    assert sorted(p.name for p in (run / "ckpt").iterdir()) == [
        "step_000025.zip",
        "step_000050.zip",
    ]
    history = load_training_history(run)
    assert history.timesteps.tolist() == [25, 50]
    assert history.actual_timesteps.tolist() == [32, 64]
    episodes = pd.read_csv(run / "episodes.csv")
    assert not episodes.empty
    assert episodes.source.isin(["real", "synthetic"]).all()
    assert episodes.loc[episodes.source == "real", "scenario"].eq(-1).all()
    assert episodes.loc[episodes.source == "synthetic", "scenario"].isin([0, 1]).all()
    before = file_hashes(run)
    with pytest.raises(FileExistsError):
        logged.main(args)
    expected = pd.DataFrame(
        checkpoints.eval_run(run, prices, test, [("validation", *valid)], 0.0025)
    )
    # Exercise real process-worker import/pickling through the formal CLI.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            checkpoints.__name__,
            "--root",
            str(root),
            "--prices",
            str(prices),
            "--test",
            *test,
            "--extra",
            "validation",
            *valid,
            "--workers",
            "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(root / "checkpoint_eval.csv")
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.timesteps.unique().tolist() == [25, 50]
    assert actual.actual_timesteps.unique().tolist() == [32, 64]
    assert set(actual.period) == {"2010", "2010-2010", "validation"}
    assert file_hashes(run) == before
