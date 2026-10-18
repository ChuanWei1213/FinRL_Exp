"""Run logged scenario-mix training for multiple datasets and seeds on CPUs.

Each worker is an independent training process with one PyTorch/BLAS thread.
All paths are relative to the invoking directory. --out must be a new directory.
--dry-run validates arguments and file locations without starting any training.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = "finrl.experiments.train_scenario_mix_logged"
THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        required=True,
        metavar=("GROUP", "NPZ"),
        help="Group name and synthetic paths.npz; repeat for multiple datasets",
    )
    ap.add_argument("--seeds", nargs="+", type=nonnegative_int, default=[0, 1, 2])
    ap.add_argument("--workers", type=positive_int, default=1)
    ap.add_argument(
        "--out", type=Path, required=True, help="New batch output directory"
    )
    ap.add_argument(
        "--p", type=finite_float, default=0.5, help="Synthetic episode share"
    )
    ap.add_argument("--train", nargs=2, default=["2010-01-01", "2017-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2018-01-01", "2019-12-31"])
    ap.add_argument("--timesteps", type=positive_int, default=500_000)
    ap.add_argument("--eval-freq", type=positive_int, default=25_000)
    ap.add_argument("--commission", type=finite_float, default=0.0025)
    ap.add_argument("--train-commission", type=finite_float, default=None)
    ap.add_argument("--k-max", type=nonnegative_int, default=21)
    ap.add_argument("--log-std-init", type=finite_float, default=-2.0)
    ap.add_argument("--ent-coef", type=finite_float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    out: Path
    log: Path


def build_jobs(cli):
    """Validate the whole batch before creating files or launching processes."""
    output = cli.out.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new --out directory: {cli.out}")
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if not 0 <= cli.p <= 1:
        raise ValueError("--p must be between 0 and 1")
    for name in ("commission", "train_commission"):
        value = getattr(cli, name)
        if value is not None and not 0 <= value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if cli.ent_coef < 0:
        raise ValueError("--ent-coef must be nonnegative")
    if cli.timesteps % cli.eval_freq:
        raise ValueError("--timesteps must be a multiple of --eval-freq")
    tr_start, tr_end, va_start, va_end = map(
        date.fromisoformat, (*cli.train, *cli.valid)
    )
    if not tr_start <= tr_end < va_start <= va_end:
        raise ValueError("Require train start <= train end < valid start <= valid end")

    prices = cli.prices.expanduser().resolve()
    if not prices.is_file():
        raise FileNotFoundError(f"Price CSV does not exist: {prices}")
    datasets = []
    names = set()
    for name, source in cli.dataset:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid dataset group name: {name!r}")
        if name.casefold() in names:
            raise ValueError(f"Duplicate dataset group name: {name}")
        names.add(name.casefold())
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Synthetic NPZ does not exist: {source}")
        datasets.append((name, source))

    shared = ["--prices", str(prices), "--train", *cli.train, "--valid", *cli.valid]
    for key in (
        "p",
        "timesteps",
        "eval_freq",
        "commission",
        "train_commission",
        "k_max",
        "log_std_init",
        "ent_coef",
    ):
        value = getattr(cli, key)
        if value is not None:
            shared.extend([f"--{key.replace('_', '-')}", str(value)])

    root = cli.out.expanduser().resolve()
    jobs = []
    for seed in cli.seeds:
        for group, source in datasets:
            name = f"{group}_seed{seed}"
            out = root / name
            command = [
                sys.executable,
                "-u",
                "-m",
                TRAINER,
                *shared,
                "--synthetic",
                str(source),
                "--group",
                group,
                "--seed",
                str(seed),
                "--out",
                str(out),
            ]
            jobs.append(Job(name, command, out, root / "logs" / f"{name}.log"))
    return jobs


def write_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def run(cli):
    jobs = build_jobs(cli)
    manifest = {
        "config": vars(cli),
        "workers": min(cli.workers, len(jobs)),
        "cwd": str(PROJECT_ROOT),
        "thread_environment": THREAD_ENV,
        "jobs": [asdict(job) for job in jobs],
    }
    if cli.dry_run:
        print(json.dumps(manifest, indent=2, default=str))
        return manifest

    root = cli.out.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    write_json(root / "manifest.json", manifest)
    status = {job.name: {"state": "queued"} for job in jobs}

    def update(job, state, **details):
        status[job.name].update(
            state=state,
            updated_at=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        write_json(root / "status.json", status)
        print(f"{job.name}: {state}", flush=True)

    write_json(root / "status.json", status)
    print(
        f"Scheduling {len(jobs)} runs with {manifest['workers']} workers: {root}",
        flush=True,
    )
    pending = deque(jobs)
    active = {}
    environment = {**os.environ, **THREAD_ENV}
    try:
        while pending or active:
            while pending and len(active) < cli.workers:
                job = pending.popleft()
                try:
                    with job.log.open("x") as log:
                        process = subprocess.Popen(
                            job.command,
                            cwd=PROJECT_ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                except OSError as error:
                    update(job, "failed", error=str(error))
                    continue
                active[process] = job
                update(job, "running", pid=process.pid)
            for process, job in list(active.items()):
                returncode = process.poll()
                if returncode is not None:
                    update(
                        job,
                        "succeeded" if returncode == 0 else "failed",
                        returncode=returncode,
                    )
                    del active[process]
            if active:
                time.sleep(0.1)
    except BaseException:
        # Stop all running children before waiting; leave partial artifacts intact.
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process, job in active.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            update(job, "cancelled", returncode=process.returncode)
        for job in jobs:
            if status[job.name]["state"] == "queued":
                update(job, "cancelled")
        raise

    failed = [name for name, item in status.items() if item["state"] == "failed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} runs failed: {', '.join(failed)}; see {root / 'logs'}"
        )
    return status


def main(argv=None):
    parser = build_parser()
    try:
        run(parser.parse_args(argv))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; running workers were stopped.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
