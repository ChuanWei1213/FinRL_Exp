"""Prepare paired real-window controls and launch 30 new runs with 5 workers.

All new outputs use dedicated directories. Original data, checkpoints, and core
training code are read-only. Re-running this entry point refuses existing outputs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
TD = ROOT.parent / "TimeDiff"
BASE = TD / "outputs/dow10/conditional/empirical_joint/run_20261017"
OUT = ROOT / "results/scenario_mix/comprehensive_20261019"
BATCH = ROOT / "results/scenario_mix/joint_missing_controls_seeds0_9_20261019_run01"
PRICES = TD / "datasets/dow10/prices_2004_2025.csv"
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-comprehensive-20261019/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/finrl-comprehensive-20261019/cache")
sys.modules.setdefault("quantstats", None)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from finrl.experiments import scenario_mix as S
from finrl.experiments import train_scenario_mix_batch as B
from finrl.experiments.scenario_mix_common import TICKERS
from analysis.scenario_mix.comprehensive_20261019.train_missing import RealReplayEnv


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def prepare():
    if (OUT.exists() and any(OUT.iterdir())) or BATCH.exists():
        raise FileExistsError("A study output already exists; do not duplicate or overwrite it")
    full = BASE / "batches/full100_20261019"
    fm = json.loads((full / "generation.json").read_text())
    assert fm["status"] == "complete" and fm["n_generated_paths"] == 2000
    assert fm["sampling"]["n_tail_draws"] == 0
    market = S.MarketData(PRICES, TICKERS)
    original = S.MarketData(TD / "datasets/dow10/prices_2004.csv", TICKERS)
    positions = market.dates.get_indexer(original.dates)
    assert (positions >= 0).all()
    overlap = {}
    for field in S.FEATURES:
        current, previous = getattr(market, field)[positions], getattr(original, field)
        overlap[field] = float(np.max(np.abs(current / previous - 1)))
        np.testing.assert_allclose(current, previous, rtol=2e-6, atol=0)
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = []
    datasets = []
    for q in [30, 10]:
        source = BASE / f"batches/full70_tail30_pr{q}/paths.npz"
        z = dict(np.load(source, allow_pickle=False))
        a = np.array([market.idx(str(d)) for d in z["anchor_dates"]])
        assert (market.dates[a] == pd.to_datetime(z["anchor_dates"])).all()
        assert (market.dates[a + 21] <= pd.Timestamp("2017-12-31")).all()
        future_indices = a[:, None] + np.arange(1, 22)
        values = {field: getattr(market, field)[future_indices] for field in S.FEATURES}
        raw = np.empty((len(a), 21, 30), dtype=np.float64)
        raw[..., ::3] = np.log(values["close"] / market.close[future_indices - 1])
        raw[..., 1::3] = np.log(values["high"] / values["close"])
        raw[..., 2::3] = np.log(values["low"] / values["close"])
        paired = z["paired_real_future"][z["draw_id"]]
        # The extended price CSV was downloaded separately; adjusted prices differ
        # by at most 1.282 ppm. Keep the PPO price source identical to the prior
        # matched runs and disclose/measure this tiny data-revision difference.
        np.testing.assert_allclose(raw, paired, rtol=0, atol=2e-6)
        rp = np.log(np.exp(raw[..., ::3]).mean(axis=-1))
        realized = np.column_stack([np.expm1(rp.sum(axis=1)), rp.std(axis=1, ddof=1)*np.sqrt(252)])
        np.testing.assert_allclose(realized, z["target"], rtol=0, atol=2e-6)
        directory = OUT / "inputs" / f"real_replay_pr{q}"
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / "paths.npz"
        keep = {k: z[k] for k in ["tickers", "anchor_dates", "scenario", "source_pool", "window_id", "draw_id", "replicate_id", "target"]}
        np.savez_compressed(destination, **keep, **values, raw_features=raw, realized=realized,
                            anchor_close=market.close[a], path_type=np.array("paired_real_replay"))
        synthetic = S.SyntheticPaths(source, market)
        replay = S.SyntheticPaths(destination, market)
        reference = S.ScenarioMixEnv(market, synthetic, p_synthetic=.5, data_start="2010-01-01", data_end="2017-12-31", commission=.0025, seed=0)
        env = RealReplayEnv(market, replay, p_synthetic=.5, data_start="2010-01-01", data_end="2017-12-31", commission=.0025, seed=0)
        eligibility = []
        for k in range(22):
            np.testing.assert_array_equal(env._syn_ok[k], reference._syn_ok[k])
            valid = env._syn_ok[k]
            eligibility.append({"k": k, "eligible_paths": len(valid), "tail_pool_fraction": float((z["scenario"][valid] == 1).mean())})
            j = int(valid[0])
            frame = env.frame_synthetic(j, k)
            for field in S.FEATURES:
                np.testing.assert_array_equal(frame[field].to_numpy().reshape(-1, 10)[-21:], values[field][j])
        for seed in range(3):
            obs, _ = env.reset(seed=seed)
            count = 0
            while True:
                obs, reward, done, truncated, _ = env.step(np.zeros(11, np.float32))
                assert np.isfinite(reward) and all(np.isfinite(v).all() for v in obs.values())
                count += 1
                if done or truncated:
                    break
            assert count == env.last_source[2] + 21
            assert env.last_source[0] in ["real", "real_replay"]
        metadata = {"group": f"REAL_REPLAY_PR{q}_B50", "path_count": len(a), "source": str(source), "source_sha256": sha(source),
                    "paths_sha256": sha(destination), "paired_real_ohlc_exact_to_ppo_prices": True,
                    "condition_max_abs_difference_vs_generator_prices": np.max(np.abs(realized-z["target"]),axis=0).tolist(),
                    "eligible_indices_match_synthetic": True, "eligibility_by_k": eligibility,
                    "source_pool_tail_fraction": float((z["scenario"]==1).mean()),
                    "interpretation": "50% ordinary real episodes; 50% same weighted anchor pool as corresponding synthetic group, replaying actual future OHLC. Duplicates preserve sampling weights; they are not independent histories."}
        write(directory / "provenance.json", metadata)
        preflight.append(metadata)
        datasets += ["--dataset", f"REAL_REPLAY_PR{q}_B50", str(destination)]
    datasets += ["--dataset", "FULL_B50", str(full / "paths.npz")]
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "train": ["2010-01-01", "2017-12-31"], "valid": ["2018-01-01", "2019-12-31"],
        "test": ["2020-01-01", "2025-12-31"], "test_subperiods": ["2020-2022", "2023-2025", "each calendar year"],
        "new_groups": ["REAL_REPLAY_PR30_B50", "REAL_REPLAY_PR10_B50", "FULL_B50"],
        "new_seeds": list(range(10)), "workers": 5, "nominal_steps": 500000, "actual_steps": 532480,
        "selection": "Maximum validation Sharpe per group and seed; earliest nominal step wins ties. Common corrected TRF and commission 0.0025. Freeze selection before any new test rollout.",
        "test_checkpoint_curves": "Descriptive diagnostics only; never used for selection, tuning, or stopping.",
        "legacy_sources": ["v2_rewardlog", "v3_nocost"],
        "legacy_reference": "Also audit the saved v2_sharpe models against corresponding v2_rewardlog selections; do not count duplicates as independent runs.",
        "legacy_differences": "Original v2/v3 training used legacy TRF; v3 training commission=0. All reevaluations use corrected TRF and commission=0.0025. Report historical validation choice sensitivity.",
        "train_metrics": "Deterministic continuous real train-window evaluation, separate from sampled stochastic training episode reward.",
        "paired_controls": "Replacement probability remains 0.5 for every B50 group. Each replay group uses exactly the same anchor/condition draws and 4x replicate weights as its generated counterpart.",
        "input_price_overlap_exact": False,
        "input_price_overlap_max_relative_difference": overlap,
        "price_revision_note": "Generator uses original prices_2004.csv; all current PPO controls and standardized evaluation use prices_2004_2025.csv, as did the existing current runs. Overlap differs by <=1.282 ppm in levels, not exactly equal.",
        "no_original_outputs_overwritten": True,
    }
    write(OUT / "protocol.json", protocol)
    write(OUT / "preflight.json", preflight)
    return datasets


def main():
    datasets = prepare()
    runtime = OUT / "runtime"
    runtime.mkdir()
    old = ROOT / "results/scenario_mix/joint_pr30_pr10_b50_seeds0_9_20261018_run02/runtime/sitecustomize.py"
    shutil.copy2(old, runtime / "sitecustomize.py")
    os.environ["PYTHONPATH"] = str(runtime)
    os.environ["FINRL_SCENARIO_DISABLE_QUANTSTATS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    B.TRAINER = "analysis.scenario_mix.comprehensive_20261019.train_missing"
    cli = B.build_parser().parse_args(["--prices", str(PRICES), *datasets, "--p", "0.5", "--seeds", *map(str, range(10)),
        "--workers", "5", "--timesteps", "500000", "--eval-freq", "25000", "--train-commission", "0.0025", "--out", str(BATCH)])
    source_files = [ROOT / "finrl/experiments/scenario_mix.py", ROOT / "finrl/experiments/train_scenario_mix_logged.py",
                    ROOT / "finrl/experiments/scenario_mix_common.py", ROOT / "finrl/experiments/notebook_utils.py",
                    ROOT / "finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py", Path(__file__), Path(__file__).with_name("train_missing.py")]
    launch = {"started_at": datetime.now(timezone.utc).isoformat(), "config": vars(cli), "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
              "input_sha256": {str(PRICES): sha(PRICES), **{path: sha(path) for _, path in cli.dataset}}, "python": sys.version,
              "runtime_note": "Same optional QuantStats report shim; no changes to PPO, reward, fees, observation, or optimizer."}
    write(runtime / "launch.json", launch)
    B.run(cli)
    write(runtime / "training_finished.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "status": "all_succeeded"})


if __name__ == "__main__":
    main()
