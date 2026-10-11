"""Paired TRF reruns. Legacy code is isolated at runtime; historical files stay intact.

Run as ``python -m analysis.scenario_mix.trf_ablation --help``.
The legacy step is read from the parent of the TRF fix, including its original dtype.
Use separate processes for training variants: the temporary class patch is not thread-safe.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# QuantStats only serves terminal plots in the base env, which scenario-mix disables.
# Keep its optional SciPy binary dependency out of these headless experiment workers.
# No training reward or evaluation metric is computed by QuantStats in this workflow.
sys.modules.setdefault("quantstats", None)

from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS, validation_score

REPO = Path(__file__).resolve().parents[2]
LEGACY_REF = "8fa237e3^"
ENV_FILE = "finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py"
GROUPS = {"A": 0.0, "B20": 0.2, "B50": 0.5, "B70": 0.7, "B90": 0.9}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def trf_version(version):
    """Switch only the base environment step; restore it even on failure."""
    from finrl.meta.env_portfolio_optimization import env_portfolio_optimization as E

    original = E.PortfolioOptimizationEnv.step
    if version == "legacy":
        source = subprocess.check_output(
            ["git", "show", f"{LEGACY_REF}:{ENV_FILE}"], cwd=REPO, text=True
        )
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == "PortfolioOptimizationEnv")
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        namespace = dict(vars(E))
        exec(compile(ast.Module(body=[step], type_ignores=[]),
                     f"git:{LEGACY_REF}:{ENV_FILE}", "exec"), namespace)
        E.PortfolioOptimizationEnv.step = namespace["step"]
    elif version != "fixed":
        raise ValueError(version)
    try:
        yield
    finally:
        E.PortfolioOptimizationEnv.step = original


def train(cli):
    from finrl.experiments import train_scenario_mix_logged as T

    args = T.build_parser().parse_args([
        "--prices", str(cli.prices), "--synthetic", str(cli.synthetic),
        "--group", cli.group, "--p", str(GROUPS[cli.group]),
        "--seed", str(cli.seed), "--out", str(cli.out),
        "--timesteps", str(cli.timesteps), "--eval-freq", str(cli.eval_freq),
    ])
    started = time.time()
    with trf_version(cli.variant):
        T.run(args)
    (cli.out / "completed.json").write_text(json.dumps({
        "variant": cli.variant, "elapsed_seconds": time.time() - started,
        "legacy_ref": LEGACY_REF, "python": sys.version,
        "prices_sha256": sha256(cli.prices), "synthetic_sha256": sha256(cli.synthetic),
    }, indent=2) + "\n")


def paired_rollout(model, market, start, end, commission=0.0025):
    """Predict once with fixed TRF, replay identical actions with original legacy step.

    TRF does not enter policy observations: prices and previous target action are
    identical. Assert this at every replay step instead of assuming it silently.
    """
    s, e = market.idx(start), market.last_idx(end)
    if s < 50 or e < s or market.dates[-1] < pd.Timestamp(end) - pd.Timedelta(days=7):
        raise ValueError(f"Insufficient market history for {start}..{end}")
    frame = S.episode_frame(market.dates[s-50:e+1], market.close[s-50:e+1],
                            market.high[s-50:e+1], market.low[s-50:e+1], market.tickers)
    outputs, actions, observations = {}, [], []
    ideal_cash = [1 / (len(market.tickers) + 1)]
    for variant in ("fixed", "legacy"):
        with trf_version(variant):
            inner = S._InnerEpisode(frame, len(market.tickers), commission, 50)
            obs, _ = inner.reset()
            obs = S.normalize_obs(obs)
            done, i = False, 0
            while not done:
                if variant == "fixed":
                    action = model.predict(obs, deterministic=True)[0]
                    actions.append(action.copy())
                    # State is solely market data; previous target action closes recurrence.
                    observations.append(obs["last_action"].copy())
                else:
                    np.testing.assert_array_equal(obs["last_action"], observations[i])
                    action = actions[i]
                obs, _, done, _, _ = inner.step(action)
                if variant == "fixed":
                    target = np.asarray(inner.env._actions_memory[-1], dtype=np.float64)
                    drifted = target * np.asarray(inner.env._price_variation, dtype=np.float64)
                    ideal_cash.append(drifted[0] / drifted.sum())
                obs = S.normalize_obs(obs)
                i += 1
            assert i == len(actions)
            env = inner.env
            n = len(market.tickers)
            finals = np.asarray(env._final_weights, float)
            acts = np.asarray(env._actions_memory, float)
            daily = pd.DataFrame({
                "date": market.dates[s-1:e+1],
                "value": np.asarray(env._asset_memory["final"], float) / env._initial_amount
                         * (1 - commission * n / (n+1)),
                "cash_weight": finals[:, 0],
                "turnover": np.r_[n/(n+1), np.abs(acts[1:] - finals[:-1]).sum(axis=1)/2],
            })
            outputs[variant] = daily
            # Legacy portfolio multiplication/normalization is float32. Several
            # roundings can exceed 2e-7 for cash-heavy policies; also check each
            # result against the independent, fee-free analytic weight identity.
            np.testing.assert_allclose(daily.cash_weight, ideal_cash, rtol=0, atol=1e-6)
            env.close()
    np.testing.assert_allclose(outputs["fixed"].cash_weight, outputs["legacy"].cash_weight,
                               rtol=0, atol=1e-6)
    return outputs


def evaluate(cli):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    run = cli.run
    config = json.loads((run / "config.json").read_text())
    market = S.MarketData(cli.prices, TICKERS)
    test_market = S.MarketData(cli.test_prices, TICKERS)
    cli.out.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted((run / "ckpt").glob("step_*.zip"))
    expected = list(range(config["eval_freq"], config["timesteps"]+1, config["eval_freq"]))
    if [int(p.stem.split("_")[1]) for p in checkpoints] != expected:
        raise ValueError(f"Missing or unexpected checkpoints in {run}")
    valid_rows, test_rows, input_rows = [], [], []
    windows = [
        ("valid", market, *config["valid"], {"valid": tuple(config["valid"])}),
        ("test", test_market, "2020-01-02", "2025-12-31", S.test_periods("2020-01-02", "2025-12-31")),
        ("gfc", test_market, "2008-01-02", "2009-06-30", {"gfc": ("2008-01-02", "2009-06-30")}),
    ]
    for ck in checkpoints:
        nominal = int(ck.stem.split("_")[1])
        model = PPO.load(ck, device="cpu")
        meta = {"group": config["group"], "seed": config["seed"], "timesteps": nominal,
                "actual_timesteps": int(model.num_timesteps)}
        input_rows.append({**meta, "checkpoint": str(ck.resolve()), "sha256": sha256(ck)})
        for name, m, start, end, periods in windows:
            paired = paired_rollout(model, m, start, end, config["commission"])
            for variant, daily in paired.items():
                if name == "valid":
                    fv = float(daily.value.iloc[-1])
                    valid_rows.append({**meta, "trf": variant, "valid_final_value": fv,
                        "valid_reward": 100*np.log(fv), "valid_sharpe": validation_score(daily, "sharpe"),
                        "valid_calmar": validation_score(daily, "calmar"),
                        "valid_mean_stock_weight": float(1-daily.cash_weight.mean())})
                else:
                    test_rows.extend({**meta, "trf": variant, **r}
                                     for r in S.period_metrics(daily, periods).to_dict("records"))
                # Daily evidence for every checkpoint supports crash-window diagnostics later.
                target = cli.out / "daily" / f"step_{nominal:06d}_{name}_{variant}.csv.gz"
                target.parent.mkdir(exist_ok=True)
                daily.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(valid_rows).to_csv(cli.out / "valid_paired.csv", index=False)
        pd.DataFrame(test_rows).to_csv(cli.out / "test_paired.csv", index=False)
        pd.DataFrame(input_rows).to_csv(cli.out / "checkpoint_manifest.csv", index=False)
        print(f"{run.parent.name}/{run.name}: evaluated {nominal} ({model.num_timesteps} actual)", flush=True)
    (cli.out / "completed.json").write_text(json.dumps({
        "run": str(run.resolve()), "checkpoints": len(checkpoints),
        "prices_sha256": sha256(cli.prices), "test_prices_sha256": sha256(cli.test_prices),
        "legacy_ref": LEGACY_REF,
    }, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--prices", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "train":
            p.add_argument("--synthetic", type=Path, required=True)
            p.add_argument("--variant", choices=["fixed", "legacy"], required=True)
            p.add_argument("--group", choices=list(GROUPS), required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--timesteps", type=int, default=500000)
            p.add_argument("--eval-freq", type=int, default=25000)
        else:
            p.add_argument("--test-prices", type=Path, required=True)
            p.add_argument("--run", type=Path, required=True)
    cli = ap.parse_args(argv)
    if cli.command == "train":
        import fcntl

        cli.out.parent.mkdir(parents=True, exist_ok=True)
        with (cli.out.parent / f".{cli.out.name}.training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = cli.out / "completed.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                if (previous["variant"] != cli.variant
                    or previous["prices_sha256"] != sha256(cli.prices)
                    or previous["synthetic_sha256"] != sha256(cli.synthetic)):
                    raise ValueError("Completed training uses different inputs")
                print(f"Already complete: {cli.out}", flush=True)
            else:
                train(cli)
    else:
        # A historical reevaluation may be requested while a batch has it queued.
        # Serialize writers and reuse only a completed result for the same inputs.
        import fcntl

        cli.out.mkdir(parents=True, exist_ok=True)
        with (cli.out / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = cli.out / "completed.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                if (previous["run"] != str(cli.run.resolve())
                    or previous["prices_sha256"] != sha256(cli.prices)
                    or previous["test_prices_sha256"] != sha256(cli.test_prices)):
                    raise ValueError("Completed evaluation uses different inputs")
                print(f"Already complete: {cli.out}", flush=True)
            else:
                evaluate(cli)


if __name__ == "__main__":
    main()
