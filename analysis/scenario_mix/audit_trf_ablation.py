"""Verify the completed TRF experiment against its inputs and checkpoint receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from analysis.scenario_mix.trf_ablation import ENV_FILE, GROUPS, REPO, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(json.loads((root / "completed_all.json").read_text())["jobs"] == 60,
            "Batch did not complete all 60 jobs")
    require(not (root / "failures.json").exists(), "Batch has failures")
    require(sha256(REPO / ENV_FILE) == manifest["env_sha256"], "Environment source changed")
    source = json.loads((root / "scientific_source_manifest.json").read_text())
    for name, digest in source["sha256"].items():
        if name.startswith("finrl/"):
            require(sha256(REPO / name) == digest, f"Scientific source changed: {name}")

    data = {}
    for name in ["prices", "test_prices", "synthetic"]:
        path = Path(manifest[name]["path"])
        require(sha256(path) == manifest[name]["sha256"], f"Input changed: {name}")
        if name != "synthetic":
            df = pd.read_csv(path)
            require(not df.isna().any().any(), f"Null values in {name}")
            require(not df.duplicated(["date", "tic"]).any(), f"Duplicate keys in {name}")
            data[name] = {"rows": len(df), "stocks": df.tic.nunique(),
                          "start": df.date.min(), "end": df.date.max()}
        else:
            with np.load(path, allow_pickle=False) as a:
                for field in ["close", "high", "low"]:
                    require(np.isfinite(a[field]).all(), f"Non-finite paths: {field}")
                scenarios, counts = np.unique(a["scenario"], return_counts=True)
                data[name] = {"shape": list(a["close"].shape),
                              "scenario_counts": dict(zip(map(str, scenarios), map(int, counts)))}

    labels = ["v2_rewardlog", "v3_nocost", "paired_legacy", "paired_fixed"]
    nominal = np.arange(25000, 500001, 25000)
    actual = np.arange(1, 21) * 26624
    checkpoint_count, daily_count, max_daily_weight_error = 0, 0, 0.0
    training, validation_reproduction = [], []
    for label in labels:
        for group in GROUPS:
            for seed in range(3):
                evaluated = root / "evaluations" / label / f"{group}_seed{seed}"
                marker = json.loads((evaluated / "completed.json").read_text())
                require(marker["checkpoints"] == 20, f"Wrong checkpoint count: {evaluated}")
                for name in ["prices", "test_prices"]:
                    require(marker[name + "_sha256"] == manifest[name]["sha256"],
                            f"Evaluation input mismatch: {evaluated}")
                receipts = pd.read_csv(evaluated / "checkpoint_manifest.csv")
                np.testing.assert_array_equal(receipts.timesteps, nominal)
                np.testing.assert_array_equal(receipts.actual_timesteps, actual)
                for row in receipts.itertuples(index=False):
                    ck = Path(row.checkpoint)
                    require(sha256(ck) == row.sha256, f"Checkpoint changed: {ck}")
                    with zipfile.ZipFile(ck) as z:
                        require(z.testzip() is None, f"Damaged checkpoint: {ck}")
                        metadata = json.loads(z.read("data"))
                        if label.startswith("paired_"):
                            runtime = z.read("system_info.txt").decode()
                            require(f"- PyTorch: {manifest['versions']['torch']}\n" in runtime,
                                    f"Training runtime mismatch: {ck}")
                    require(metadata["num_timesteps"] == row.actual_timesteps,
                            f"Checkpoint step mismatch: {ck}")
                    require(metadata["seed"] == seed, f"Checkpoint seed mismatch: {ck}")
                    checkpoint_count += 1
                daily = list((evaluated / "daily").glob("*.csv.gz"))
                require(len(daily) == 120, f"Missing daily evidence: {evaluated}")
                daily_count += len(daily)
                for fixed_path in (evaluated / "daily").glob("*_fixed.csv.gz"):
                    legacy_path = fixed_path.with_name(fixed_path.name.replace("_fixed.csv.gz", "_legacy.csv.gz"))
                    a, b = (pd.read_csv(p, usecols=["date", "cash_weight"]) for p in [fixed_path, legacy_path])
                    require(a.date.equals(b.date), f"Unpaired daily dates: {fixed_path}")
                    error = float((a.cash_weight - b.cash_weight).abs().max())
                    require(error <= 1e-6, f"Holdings differ beyond float32 rounding: {fixed_path}")
                    max_daily_weight_error = max(max_daily_weight_error, error)
                if label.startswith("paired_"):
                    run = root / label / f"{group}_seed{seed}"
                    done = json.loads((run / "completed.json").read_text())
                    config = json.loads((run / "config.json").read_text())
                    require(done["python"] == manifest["python"], f"Training Python mismatch: {run}")
                    require(done["variant"] == label.removeprefix("paired_"), f"Wrong variant: {run}")
                    require(config["group"] == group and config["seed"] == seed, f"Wrong config: {run}")
                    require(config["p"] == GROUPS[group], f"Wrong mix ratio: {run}")
                    require(config["commission"] == .0025 and config["train_commission"] is None,
                            f"Wrong training cost: {run}")
                    require(config["timesteps"] == 500000 and config["eval_freq"] == 25000,
                            f"Wrong training budget: {run}")
                    for name in ["prices", "synthetic"]:
                        require(done[name + "_sha256"] == manifest[name]["sha256"],
                                f"Training input mismatch: {run}")
                    valid = pd.read_csv(run / "valid.csv")
                    np.testing.assert_array_equal(valid.actual_timesteps, actual)
                    np.testing.assert_array_equal(valid.timesteps, nominal)
                    reevaluated = pd.read_csv(evaluated / "valid_paired.csv")
                    reevaluated = reevaluated[reevaluated.trf == done["variant"]]
                    joined = valid.merge(reevaluated, on=["timesteps", "actual_timesteps"],
                                         suffixes=("_training", "_rerun"), validate="one_to_one")
                    for metric in ["valid_final_value", "valid_reward", "valid_sharpe",
                                   "valid_calmar", "valid_mean_stock_weight"]:
                        diff = (joined[metric + "_training"] - joined[metric + "_rerun"]).abs()
                        require(diff.max() <= 1e-12, f"Training validation not reproduced: {run} {metric}")
                        validation_reproduction.append({"label": label, "group": group,
                            "seed": seed, "metric": metric, "checkpoints": len(joined),
                            "max_abs_error": diff.max()})
                    episodes = pd.read_csv(run / "episodes.csv")
                    np.testing.assert_array_equal(episodes.length.cumsum(), episodes.timesteps)
                    require(episodes.timesteps.iloc[-1] <= 532480, f"Episode beyond budget: {run}")
                    training.append({"label": label, "group": group, "seed": seed,
                                     "episodes": len(episodes), "actual_timesteps": 532480})

    report = root / "report"
    validation = pd.read_csv(report / "validation.csv")
    evaluations = pd.read_csv(report / "evaluations.csv")
    require(len(validation) == 2400 and len(evaluations) == 19200, "Incomplete final report")
    for df in [validation, evaluations]:
        require(np.isfinite(df.select_dtypes("number")).all().all(), "Non-finite report metrics")
    pairing = pd.read_csv(report / "episode_pairing.csv")
    require(len(pairing) == 15 and pairing.same_episode_draws.all(), "Unpaired episode draws")
    for group in GROUPS:
        for seed in range(3):
            configs = []
            for label in ["paired_legacy", "paired_fixed"]:
                config = json.loads((root / label / f"{group}_seed{seed}" / "config.json").read_text())
                config.pop("out")
                configs.append(config)
            require(configs[0] == configs[1], f"Unpaired training settings: {group} seed {seed}")
    pd.DataFrame(validation_reproduction).to_csv(root / "training_validation_reproduction.csv", index=False)
    receipt = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "data": data, "checkpoints_verified": checkpoint_count,
               "daily_files": daily_count, "training_runs": len(training),
               "max_daily_cash_weight_difference": max_daily_weight_error,
               "paired_episode_sequences": len(pairing), "validation_rows": len(validation),
               "evaluation_rows": len(evaluations), "training": training}
    (root / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["data", "training"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    audit(ap.parse_args().root)


if __name__ == "__main__":
    main()
