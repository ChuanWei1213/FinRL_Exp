"""Evaluate every saved checkpoint of log_training_rewards.py runs on real data. No training.

    python examples/evaluate_checkpoints.py --root results/scenario_mix/v3_nocost \
        --prices ../TimeDiff/datasets/dow10/prices_2004_2025.csv

For each <root>/<group>_seed<k>/ckpt/step_<t>.zip: one continuous deterministic run on the
test window (yearly + full-period metrics) and on each --extra window (default: the 2008-09
crisis, which lies before the 2010 training start). Commission is always charged.
Output: <root>/checkpoint_eval.csv  (group, seed, timesteps, period, metrics...)
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finrl.experiments import scenario_mix as S  # noqa: E402

TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]


def eval_run(run: Path, prices: Path, test, extra, commission):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    g, seed = re.match(r"(.+)_seed(\d+)$", run.name).groups()
    market = S.MarketData(prices, TICKERS)
    periods = S.test_periods(*test)
    rows = []
    for ck in sorted((run / "ckpt").glob("step_*.zip")):
        t = int(ck.stem.split("_")[1])
        model = PPO.load(ck, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
        windows = [(periods, test)] + [({name: (a, b)}, (a, b)) for name, a, b in extra]
        for per, (a, b) in windows:
            daily = S.run_continuous(policy, market, a, b, commission)
            for r in S.period_metrics(daily, per).to_dict("records"):
                rows.append({"group": g, "seed": int(seed), "timesteps": t, **r})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--test", nargs=2, default=["2020-01-02", "2025-12-31"])
    ap.add_argument("--extra", nargs=3, action="append", metavar=("NAME", "START", "END"),
                    default=None, help="extra evaluation window (default: gfc 2008-01-02 2009-06-30)")
    ap.add_argument("--commission", type=float, default=0.0025)
    ap.add_argument("--workers", type=int, default=15)
    cli = ap.parse_args()
    extra = cli.extra if cli.extra is not None else [("gfc", "2008-01-02", "2009-06-30")]
    runs = sorted(p for p in cli.root.iterdir() if p.is_dir() and (p / "ckpt").is_dir())
    with ProcessPoolExecutor(cli.workers) as ex:
        futs = [ex.submit(eval_run, r, cli.prices, cli.test, extra, cli.commission) for r in runs]
        rows = [row for f in futs for row in f.result()]
    out = pd.DataFrame(rows)
    out.to_csv(cli.root / "checkpoint_eval.csv", index=False)
    print(f"{len(runs)} runs, {out.timesteps.nunique()} steps -> {cli.root / 'checkpoint_eval.csv'}")


if __name__ == "__main__":
    main()
