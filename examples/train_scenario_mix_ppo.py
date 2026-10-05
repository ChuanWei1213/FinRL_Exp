"""PPO, 10 stocks + cash: real only vs real + TimeDiff stress scenarios.

Groups (identical PPO / reward / fees / timesteps / seeds / model selection):
    A    p_synthetic = 0.0
    B20  p_synthetic = 0.2
    B50  p_synthetic = 0.5
Training episodes: 50 obs + k real + 21 future days, k ~ U{0..21}, start weights 1/11.
Model selection: 2018-2019 continuous real run, highest final value (same rule for all).
Test: 2020-2022 continuous real run, start 1/11, build-up cost charged once.

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
TEST_PERIODS = {"2020": ("2020-01-01", "2020-12-31"), "2021": ("2021-01-01", "2021-12-31"),
                "2022": ("2022-01-01", "2022-12-31"), "2020-2022": ("2020-01-01", "2022-12-31")}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", nargs=2, default=["2009-04-01", "2017-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2018-01-01", "2019-12-31"])
    ap.add_argument("--test", nargs=2, default=["2020-01-01", "2022-12-31"])
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--commission", type=float, default=0.0025)
    ap.add_argument("--k-max", type=int, default=21)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="Tiny networks run fastest single-threaded; >1 mostly adds contention")
    return ap.parse_args()


def make_ppo(env, seed, device):
    from stable_baselines3 import PPO
    from finrl.experiments.notebook_utils import EIIEExtractor
    return PPO("MultiInputPolicy", env, seed=seed, device=device, verbose=0,
               learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99,
               gae_lambda=0.95, ent_coef=0.0,
               policy_kwargs={"log_std_init": -2.0, "features_extractor_class": EIIEExtractor,
                              "features_extractor_kwargs": {"k_size": 3, "conv_mid": 2, "conv_final": 20}})


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

    market = S.MarketData(cli.prices, TICKERS)
    synthetic = S.SyntheticPaths(cli.synthetic, market)
    (cli.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}, indent=2) + "\n")

    # Baselines (deterministic, no training).
    rows = []
    for name, cash in (("buy_hold_1_11", True), ("buy_hold_1_10", False)):
        bh = S.buy_and_hold(market, *cli.test, cli.commission, include_cash=cash)
        bh.to_csv(cli.out / f"{name}_test_daily.csv", index=False)
        for r in S.period_metrics(bh, TEST_PERIODS).to_dict("records"):
            rows.append({"group": name, "seed": None, **r})

    from stable_baselines3 import PPO
    for group in cli.groups:
        for seed in cli.seeds:
            env = S.ScenarioMixEnv(market, synthetic, p_synthetic=GROUPS[group],
                                   data_start=cli.train[0], data_end=cli.train[1],
                                   commission=cli.commission, k_max=cli.k_max, seed=seed)
            model = make_ppo(env, seed, cli.device)
            policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
            best_path, best_val, history = cli.out / f"{group}_seed{seed}.zip", -np.inf, []
            for done in range(cli.eval_freq, cli.timesteps + 1, cli.eval_freq):
                model.learn(cli.eval_freq, reset_num_timesteps=False)
                val = S.run_continuous(policy, market, *cli.valid, cli.commission).value.iloc[-1]
                history.append({"timesteps": done, "valid_final_value": val})
                if val > best_val:
                    best_val = val
                    model.save(best_path)
            pd.DataFrame(history).to_csv(cli.out / f"{group}_seed{seed}_valid.csv", index=False)

            model = PPO.load(best_path, device=cli.device)
            daily = S.run_continuous(policy, market, *cli.test, cli.commission)
            daily.to_csv(cli.out / f"{group}_seed{seed}_test_daily.csv", index=False)
            for r in S.period_metrics(daily, TEST_PERIODS).to_dict("records"):
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
