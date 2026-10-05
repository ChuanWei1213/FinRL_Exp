"""Re-test already-trained scenario-mix PPO models on a new test period. No retraining.

The checkpoint of every run was chosen on 2018-2019 validation, which is unaffected by
the test period, so the selected models can be evaluated on a longer test window directly.

Run from FinRL_Exp/:
    python examples/evaluate_scenario_mix.py \
        --runs results/scenario_mix/v1 \
        --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv \
        --test 2020-01-02 2025-12-31 \
        --out results/scenario_mix/v1_test2025

Adding runs later (e.g. new groups) into an existing output folder:
    python examples/evaluate_scenario_mix.py ... --out results/scenario_mix/v1_test2025 \
        --only B70_seed0 B70_seed1 B70_seed2 B90_seed0 B90_seed1 B90_seed2

Output mirrors the training layout, so summarize_scenario_mix.py and the notebook read it:
    <out>/<group>_seed<k>/{config.json, summary.csv, <group>_seed<k>_test_daily.csv,
                           <group>_seed<k>_valid.csv (copied), buy_hold_*_test_daily.csv}
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments import scenario_mix as S  # noqa: E402

TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="folder with <group>_seed<k>/ run dirs")
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--test", nargs=2, default=["2020-01-02", "2025-12-31"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="+", default=None,
                    help="evaluate only these run folders, e.g. B70_seed0 B70_seed1 (default: all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    cli = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(cli.torch_threads)

    market = S.MarketData(cli.prices, TICKERS)
    if market.dates[-1] < pd.Timestamp(cli.test[1]) - pd.Timedelta(days=7):
        raise ValueError(f"prices end {market.dates[-1].date()} before test end {cli.test[1]}")
    periods = S.test_periods(*cli.test)

    runs = sorted(p for p in cli.runs.iterdir() if p.is_dir() and (p / "config.json").exists())
    if cli.only:
        runs = [p for p in runs if p.name in set(cli.only)]
    clash = [p.name for p in runs if (cli.out / p.name).exists()]
    if clash:
        raise FileExistsError(f"already evaluated in {cli.out}: {clash}; use --only to pick new runs")
    if not runs:
        raise ValueError(f"no runs found in {cli.runs}")
    baselines = []
    for name, cash in (("buy_hold_1_11", True), ("buy_hold_1_10", False)):
        bh = S.buy_and_hold(market, *cli.test, json.loads((runs[0] / "config.json").read_text())["commission"],
                            include_cash=cash)
        baselines.append((name, bh, [{"group": name, "seed": None, **r}
                                     for r in S.period_metrics(bh, periods).to_dict("records")]))

    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        if pd.Timestamp(cfg["valid"][1]) >= pd.Timestamp(cli.test[0]):
            raise ValueError(f"{run.name}: validation period overlaps the new test period")
        group, seed = cfg["groups"][0], cfg["seeds"][0]
        model_path = run / f"{group}_seed{seed}.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = PPO.load(model_path, device=cli.device)
        policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
        daily = S.run_continuous(policy, market, *cli.test, cfg["commission"])

        dst = cli.out / run.name
        dst.mkdir(parents=True)
        new_cfg = {**cfg, "test": list(cli.test), "prices_test": str(cli.prices), "out": str(dst),
                   "evaluated_from": str(model_path)}
        (dst / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
        daily.to_csv(dst / f"{group}_seed{seed}_test_daily.csv", index=False)
        valid = run / f"{group}_seed{seed}_valid.csv"
        if valid.exists():
            shutil.copy(valid, dst / valid.name)
        rows = [{"group": group, "seed": seed, **r} for r in S.period_metrics(daily, periods).to_dict("records")]
        for name, bh, bh_rows in baselines:
            bh.to_csv(dst / f"{name}_test_daily.csv", index=False)
            rows = bh_rows + rows
        pd.DataFrame(rows).to_csv(dst / "summary.csv", index=False)
        full = [r for r in rows if r["group"] == group and "-" in str(r["period"])][0]
        print(f"{run.name}: {full['period']} return {full['return']:+.1%}  max DD {full['max_drawdown']:+.1%}", flush=True)
    print(f"done -> {cli.out}")


if __name__ == "__main__":
    main()