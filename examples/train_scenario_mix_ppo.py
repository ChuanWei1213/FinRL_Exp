"""PPO, 10 stocks + cash: real only vs real + TimeDiff stress scenarios.

Groups (identical PPO / reward / fees / timesteps / seeds / model selection):
    A    p_synthetic = 0.0
    B20  p_synthetic = 0.2
    B50  p_synthetic = 0.5
Training episodes: 50 obs + k real + 21 future days, k ~ U{0..21}, start weights 1/11.
Model selection: 2018-2019 continuous real run, highest final value (same rule for all).
Test: 2020-01-02 .. 2025-12-31 continuous real run (yearly + full-period metrics), start 1/11, build-up cost charged once.

Run from FinRL_Exp/:
    python examples/train_scenario_mix_ppo.py \
        --prices ../TimeDiff/datasets/dow10/prices.csv \
        --synthetic ../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz \
        --out results/scenario_mix/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments import scenario_mix as S  # noqa: E402

TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]
GROUPS = {"A": 0.0, "B20": 0.2, "B50": 0.5}
DEFAULT_TEST = ["2020-01-02", "2025-12-31"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", nargs=2, default=["2009-04-01", "2017-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2018-01-01", "2019-12-31"])
    ap.add_argument("--test", nargs=2, default=DEFAULT_TEST)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS),
                    help="Built-in A/B20/B50 and/or names defined with --custom-group")
    ap.add_argument("--custom-group", nargs=2, action="append", default=[], metavar=("NAME", "P"),
                    help="Extra group with synthetic share P, e.g. --custom-group S99 0.99")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--commission", type=float, default=0.0025)
    ap.add_argument("--k-max", type=int, default=21)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-std-init", type=float, default=-2.0,
                    help="PPO action noise (log std). -2 = v1 setting; try -0.5 or 0 for more exploration")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--select-metric", choices=["final_value", "sharpe", "calmar"], default="final_value",
                    help="Validation rule for picking the checkpoint (same for every group)")
    ap.add_argument("--tensorboard", type=Path, default=None, help="Optional tensorboard log dir")
    ap.add_argument("--crash-diag-n", type=int, default=100,
                    help="Episodes per source for the crash-response diagnostic (0 = skip)")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="Tiny networks run fastest single-threaded; >1 mostly adds contention")
    return ap.parse_args()


def make_ppo(env, seed, device, log_std_init=-2.0, ent_coef=0.0, tensorboard=None):
    from stable_baselines3 import PPO
    from finrl.experiments.notebook_utils import EIIEExtractor
    return PPO("MultiInputPolicy", env, seed=seed, device=device, verbose=0,
               learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99,
               gae_lambda=0.95, ent_coef=ent_coef,
               tensorboard_log=str(tensorboard) if tensorboard else None,
               policy_kwargs={"log_std_init": log_std_init, "features_extractor_class": EIIEExtractor,
                              "features_extractor_kwargs": {"k_size": 3, "conv_mid": 2, "conv_final": 20}})


def validation_score(daily, metric):
    v = daily.value.to_numpy()
    if metric == "final_value":
        return float(v[-1])
    logr = np.diff(np.log(v))
    if metric == "sharpe":
        return float(logr.mean() / logr.std(ddof=1) * np.sqrt(252)) if logr.std() > 0 else -np.inf
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    ann = float(np.exp(logr.mean() * 252) - 1)
    return ann / abs(mdd) if mdd < 0 else np.inf


def main():
    cli = parse()
    import torch
    torch.set_num_threads(cli.torch_threads)
    if cli.out.exists() and any(cli.out.iterdir()):
        raise FileExistsError(f"{cli.out} is not empty")
    cli.out.mkdir(parents=True, exist_ok=True)
    tr_end, va_start, va_end, te_start = map(pd.Timestamp, (cli.train[1], cli.valid[0], cli.valid[1], cli.test[0]))
    if not tr_end < va_start <= va_end < te_start:
        raise ValueError("need train < valid < test")

    groups = dict(GROUPS)
    for name, p in cli.custom_group:
        groups[name] = float(p)
    unknown = [g for g in cli.groups if g not in groups]
    if unknown:
        raise ValueError(f"unknown groups {unknown}; define them with --custom-group")
    market = S.MarketData(cli.prices, TICKERS)
    if market.dates[-1] < pd.Timestamp(cli.test[1]) - pd.Timedelta(days=7):
        raise ValueError(f"prices end {market.dates[-1].date()} before test end {cli.test[1]}; download newer data")
    periods = S.test_periods(*cli.test)
    synthetic = S.SyntheticPaths(cli.synthetic, market)
    (cli.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}, indent=2) + "\n")

    # Baselines (deterministic, no training).
    rows = []
    for name, cash in (("buy_hold_1_11", True), ("buy_hold_1_10", False)):
        bh = S.buy_and_hold(market, *cli.test, cli.commission, include_cash=cash)
        bh.to_csv(cli.out / f"{name}_test_daily.csv", index=False)
        for r in S.period_metrics(bh, periods).to_dict("records"):
            rows.append({"group": name, "seed": None, **r})

    from stable_baselines3 import PPO
    for group in cli.groups:
        for seed in cli.seeds:
            env = S.ScenarioMixEnv(market, synthetic, p_synthetic=groups[group],
                                   data_start=cli.train[0], data_end=cli.train[1],
                                   commission=cli.commission, k_max=cli.k_max, seed=seed)
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
            model = make_ppo(env, seed, cli.device, cli.log_std_init, cli.ent_coef, cli.tensorboard)
            policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
            best_path, best_val, history = cli.out / f"{group}_seed{seed}.zip", -np.inf, []
            for done in range(cli.eval_freq, cli.timesteps + 1, cli.eval_freq):
                model.learn(cli.eval_freq, reset_num_timesteps=False, tb_log_name=f"{group}_seed{seed}")
                vdaily = S.run_continuous(policy, market, *cli.valid, cli.commission)
                val = validation_score(vdaily, cli.select_metric)
                history.append({"timesteps": done, "valid_final_value": float(vdaily.value.iloc[-1]),
                                "valid_score": val, "valid_mean_stock_weight": float(1 - vdaily.cash_weight.mean()),
                                "valid_stock_weight_std": float((1 - vdaily.cash_weight).std())})
                if val > best_val:
                    best_val = val
                    model.save(best_path)
            pd.DataFrame(history).to_csv(cli.out / f"{group}_seed{seed}_valid.csv", index=False)

            model = PPO.load(best_path, device=cli.device)
            if cli.crash_diag_n:
                diag_env = env.unwrapped if hasattr(env, "unwrapped") else env
                cr = S.crash_response(policy, diag_env, n_per_scenario=cli.crash_diag_n, seed=seed)
                cr.insert(0, "seed", seed)
                cr.insert(0, "group", group)
                cr.to_csv(cli.out / f"{group}_seed{seed}_crash_response.csv", index=False)
                print(cr.round(4).to_string(index=False), flush=True)
            daily = S.run_continuous(policy, market, *cli.test, cli.commission)
            daily.to_csv(cli.out / f"{group}_seed{seed}_test_daily.csv", index=False)
            for r in S.period_metrics(daily, periods).to_dict("records"):
                rows.append({"group": group, "seed": seed, **r})
            print(f"done {group} seed {seed}: valid {best_val:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(cli.out / "summary.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(cli.out / "summary.csv", index=False)
    agg = summary.drop(columns="seed").groupby(["period", "group"]).agg(["mean", "std"])
    agg.to_csv(cli.out / "summary_by_group.csv")
    print(summary.drop(columns="seed").groupby(["period", "group"]).mean().round(4).to_string())


if __name__ == "__main__":
    main()
