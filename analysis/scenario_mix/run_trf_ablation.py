"""Run historical reevaluation and paired full-budget retraining in fresh processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from analysis.scenario_mix.trf_ablation import ENV_FILE, GROUPS, LEGACY_REF, REPO, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--test-prices", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage", choices=["all", "historical", "training"], default="all")
    cli = ap.parse_args()
    out = cli.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "legacy_ref": subprocess.check_output(["git", "rev-parse", LEGACY_REF], text=True).strip(),
        "python": sys.version, "executable": sys.executable,
        "env_sha256": sha256(REPO / ENV_FILE),
        "prices": {"path": str(cli.prices.resolve()), "sha256": sha256(cli.prices)},
        "test_prices": {"path": str(cli.test_prices.resolve()), "sha256": sha256(cli.test_prices)},
        "synthetic": {"path": str(cli.synthetic.resolve()), "sha256": sha256(cli.synthetic)},
        "protocol": {"groups": GROUPS, "seeds": [0, 1, 2], "nominal_steps": 500000,
                     "actual_steps": 532480, "eval_freq": 25000, "commission": 0.0025,
                     "quantstats": "optional terminal reports disabled"},
    }
    import numpy, pandas, torch, stable_baselines3
    manifest["versions"] = {m.__name__: m.__version__ for m in [numpy, pandas, torch, stable_baselines3]}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Manifest differs: use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for seed in range(3):
        for group in GROUPS:
            if cli.stage != "historical":
                for variant in ("legacy", "fixed"):
                    jobs.append((f"paired_{variant}", group, seed, variant))
            if cli.stage != "training":
                for label in ("v2_rewardlog", "v3_nocost"):
                    jobs.append((label, group, seed, None))
    lock, status = threading.Lock(), {}

    def update(name, state):
        with lock:
            status[name] = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat()}
            (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(name, state, flush=True)

    def execute(job):
        label, group, seed, variant = job
        name = f"{label}/{group}_seed{seed}"
        model_root = out / label if variant else REPO / "results/scenario_mix" / label
        run = model_root / f"{group}_seed{seed}"
        target = out / "evaluations" / label / run.name
        base = [sys.executable, "-u", "-m", "analysis.scenario_mix.trf_ablation"]
        update(name, "running")
        try:
            with (logs / f"{label}_{run.name}.log").open("a") as log:
                if variant and not (run / "completed.json").exists():
                    subprocess.run(base + ["train", "--variant", variant, "--group", group,
                        "--seed", str(seed), "--prices", str(cli.prices.resolve()),
                        "--synthetic", str(cli.synthetic.resolve()), "--out", str(run)],
                        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                if not (target / "completed.json").exists():
                    subprocess.run(base + ["evaluate", "--run", str(run),
                        "--prices", str(cli.prices.resolve()), "--test-prices", str(cli.test_prices.resolve()),
                        "--out", str(target)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            update(name, "complete")
        except Exception as error:
            update(name, f"failed: {error}")
            raise

    failures = []
    with ThreadPoolExecutor(max_workers=cli.workers) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as error:
                failures.append({"job": futures[f], "error": str(error)})
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} jobs failed; see logs")
    (out / f"completed_{cli.stage}.json").write_text(json.dumps({"jobs": len(jobs)}) + "\n")


if __name__ == "__main__":
    main()
