from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from finrl.experiments import train_scenario_mix_batch as batch


@pytest.fixture
def cli_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data with spaces"
    data.mkdir()
    for name in ("prices.csv", "pr30.npz", "pr10.npz"):
        (data / name).write_text("Test path only; no PPO input is loaded.\n")
    return [
        "--prices",
        "data with spaces/prices.csv",
        "--dataset",
        "PR30_B50",
        "data with spaces/pr30.npz",
        "--dataset",
        "PR10_B50",
        "data with spaces/pr10.npz",
        "--out",
        "batch output",
    ]


@pytest.mark.parametrize("override", [[], ["--train-commission", "0", "--k-max", "10"]])
def test_commands_preserve_logged_protocol_and_relative_paths(
    cli_args, override, tmp_path
):
    from finrl.experiments import train_scenario_mix_logged as logged

    cli = batch.build_parser().parse_args(cli_args + override)
    jobs = batch.build_jobs(cli)
    assert len(jobs) == 6
    assert len({job.out for job in jobs}) == 6
    assert [job.name for job in jobs[:2]] == ["PR30_B50_seed0", "PR10_B50_seed0"]
    for job in jobs:
        assert job.command[:4] == [sys.executable, "-u", "-m", batch.TRAINER]
        child = logged.build_parser().parse_args(job.command[4:])
        original = logged.build_parser().parse_args(
            [
                "--prices",
                str(child.prices),
                "--synthetic",
                str(child.synthetic),
                "--group",
                child.group,
                "--seed",
                str(child.seed),
                "--p",
                "0.5",
                "--out",
                str(job.out),
                *override,
            ]
        )
        assert vars(child) == vars(original)
        assert child.prices == tmp_path / "data with spaces/prices.csv"
        assert child.synthetic.is_file()
        assert child.out.parent == tmp_path / "batch output"
    assert not cli.out.exists()


def test_dry_run_cli_creates_no_files_or_children(cli_args, tmp_path):
    # Invoke the real CLI from a foreign cwd with relative paths and no PPO execution.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            batch.__name__,
            *cli_args,
            "--workers",
            "3",
            "--dry-run",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(batch.PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["workers"] == 3
    assert len(manifest["jobs"]) == 6
    assert not (tmp_path / "batch output").exists()


@pytest.mark.parametrize(
    "args", [["--workers", "0"], ["--workers", "-2"], ["--p", "nan"]]
)
def test_reject_invalid_cli_values(cli_args, args):
    with pytest.raises(SystemExit):
        batch.build_parser().parse_args(cli_args + args)


@pytest.mark.parametrize(
    "args,exception",
    [
        (["--seeds", "1", "1"], ValueError),
        (["--dataset", "pr30_b50", "data with spaces/pr30.npz"], ValueError),
        (["--dataset", "../escape", "data with spaces/pr30.npz"], ValueError),
        (["--dataset", "missing", "absent.npz"], FileNotFoundError),
        (["--prices", "absent.csv"], FileNotFoundError),
        (["--timesteps", "1"], ValueError),
        (["--p", "1.5"], ValueError),
        (["--commission", "1"], ValueError),
        (["--valid", "2017-01-01", "2019-12-31"], ValueError),
    ],
)
def test_preflight_rejects_whole_batch_without_writes(cli_args, args, exception):
    cli = batch.build_parser().parse_args(cli_args + args)
    with pytest.raises(exception):
        batch.run(cli)
    assert not cli.out.exists()


def test_existing_results_are_not_overwritten(cli_args):
    cli = batch.build_parser().parse_args(cli_args)
    cli.out.mkdir()
    marker = cli.out / "original.zip"
    marker.write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        batch.run(cli)
    assert list(cli.out.iterdir()) == [marker]
    assert marker.read_bytes() == b"existing model"


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Exercise actual process scheduling without importing or training PPO."""
    script = tmp_path / "fake worker.py"
    script.write_text(
        """import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path)
p.add_argument('--group')
p.add_argument('--seed', type=int)
args, remaining = p.parse_known_args()
args.out.mkdir()
record = {
    'pid': os.getpid(), 'start': time.monotonic_ns(),
    'group': args.group, 'seed': args.seed,
    'threads': {key: os.environ.get(key) for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')},
}
print('worker stdout', flush=True)
print('worker stderr', file=sys.stderr, flush=True)
time.sleep(float(os.environ.get('FAKE_WORKER_DELAY', '0.5')))
record['end'] = time.monotonic_ns()
(args.out / 'record.json').write_text(json.dumps(record))
sys.exit(9 if args.group == 'bad' else 0)
"""
    )
    real_build = batch.build_jobs

    def fake_jobs(cli):
        return [
            replace(job, command=[sys.executable, str(script), *job.command[4:]])
            for job in real_build(cli)
        ]

    monkeypatch.setattr(batch, "build_jobs", fake_jobs)


@pytest.mark.parametrize("workers", [1, 2])
def test_real_process_concurrency_logs_and_thread_limits(
    cli_args, fake_worker, monkeypatch, workers
):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    cli = batch.build_parser().parse_args(
        cli_args + ["--workers", str(workers), "--seeds", "0", "1"]
    )
    result = batch.run(cli)
    assert all(item["state"] == "succeeded" for item in result.values())
    assert all(item["returncode"] == 0 for item in result.values())
    assert json.loads((cli.out / "status.json").read_text()) == result
    manifest = json.loads((cli.out / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 4
    records = [json.loads(path.read_text()) for path in cli.out.glob("*/record.json")]
    assert len(records) == 4
    assert len({record["pid"] for record in records}) == 4
    events = []
    for record in records:
        assert record["threads"] == batch.THREAD_ENV
        events.extend([(record["start"], 1), (record["end"], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak == workers
    assert active == 0
    assert os.environ["OMP_NUM_THREADS"] == "8"
    logs = list((cli.out / "logs").glob("*.log"))
    assert len(logs) == 4
    assert all(
        "worker stdout" in path.read_text() and "worker stderr" in path.read_text()
        for path in logs
    )


def test_failure_is_reported_and_other_jobs_finish(cli_args, fake_worker):
    args = cli_args + [
        "--dataset",
        "bad",
        "data with spaces/pr30.npz",
        "--seeds",
        "0",
        "--workers",
        "2",
    ]
    with pytest.raises(SystemExit) as failure:
        batch.main(args)
    assert failure.value.code == 1
    status = json.loads(Path("batch output/status.json").read_text())
    assert status["bad_seed0"]["state"] == "failed"
    assert status["bad_seed0"]["returncode"] == 9
    assert status["PR30_B50_seed0"]["state"] == "succeeded"
    assert status["PR10_B50_seed0"]["state"] == "succeeded"


def test_interrupt_stops_children_and_cancels_pending_jobs(
    cli_args, fake_worker, monkeypatch
):
    monkeypatch.setenv("FAKE_WORKER_DELAY", "30")
    processes = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch.subprocess, "Popen", capture)
    monkeypatch.setattr(batch, "time", SimpleNamespace(sleep=interrupt))
    with pytest.raises(SystemExit) as interruption:
        batch.main(cli_args + ["--workers", "2"])
    assert interruption.value.code == 130
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)
    status = json.loads(Path("batch output/status.json").read_text())
    assert len(status) == 6
    assert all(item["state"] == "cancelled" for item in status.values())
