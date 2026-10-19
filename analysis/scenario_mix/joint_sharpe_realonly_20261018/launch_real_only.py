"""Launch the matched real-only control without changing the training implementation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "results/scenario_mix/joint_real_only_seeds0_9_20261018_run01"
MIXED = ROOT / "results/scenario_mix/joint_pr30_pr10_b50_seeds0_9_20261018_run02"
PRICES = ROOT.parent / "TimeDiff/datasets/dow10/prices_2004_2025.csv"
SYNTHETIC = ROOT.parent / "TimeDiff/outputs/dow10/conditional/empirical_joint/run_20261017/batches/full70_tail30_pr30/paths.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if BATCH.exists():
        raise FileExistsError(f"Refusing to start a duplicate batch: {BATCH}")
    runtime = Path(tempfile.mkdtemp(prefix="finrl-realonly-20261018-", dir="/private/tmp"))
    shutil.copy2(MIXED / "runtime/sitecustomize.py", runtime / "sitecustomize.py")
    (runtime / "mpl").mkdir()
    (runtime / "cache").mkdir()
    settings = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(runtime),
        "FINRL_SCENARIO_DISABLE_QUANTSTATS": "1",
        "MPLCONFIGDIR": str(runtime / "mpl"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
    }
    settings.update({key: "1" for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]})
    environment = {**os.environ, **settings}
    preflight = r'''
import json, sys
import numpy as np
from pathlib import Path
from finrl.experiments import scenario_mix as S
from finrl.experiments.scenario_mix_common import TICKERS
market = S.MarketData(Path(sys.argv[1]), TICKERS)
synthetic = S.SyntheticPaths(Path(sys.argv[2]), market)
records = []
for seed in range(10):
    env = S.ScenarioMixEnv(market, synthetic, p_synthetic=0., data_start="2010-01-01", data_end="2017-12-31", commission=.0025, k_max=21, seed=seed)
    obs, _ = env.reset()
    steps = 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(np.zeros(11, dtype=np.float32))
        assert np.isfinite(reward)
        assert all(np.isfinite(x).all() for x in obs.values())
        steps += 1
        if terminated or truncated:
            break
    assert env.last_source[0] == "real" and env.last_source[1] is None
    assert steps == 21 + env.last_source[2]
    records.append({"seed": seed, "source": "real", "steps": steps, "passed": True})
print(json.dumps({"quantstats_disabled": sys.modules.get("quantstats") is None, "p_synthetic": 0., "episodes": records}))
'''
    checked = subprocess.run([sys.executable, "-B", "-c", preflight, str(PRICES), str(SYNTHETIC)], cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    result = json.loads(checked.stdout.strip().splitlines()[-1])
    command = [sys.executable, "-u", "-B", "-m", "finrl.experiments.train_scenario_mix_batch",
               "--prices", str(PRICES), "--dataset", "REAL_ONLY", str(SYNTHETIC),
               "--p", "0", "--seeds", *map(str, range(10)), "--workers", "5",
               "--train", "2010-01-01", "2017-12-31", "--valid", "2018-01-01", "2019-12-31",
               "--timesteps", "500000", "--eval-freq", "25000", "--commission", "0.0025",
               "--k-max", "21", "--log-std-init", "-2", "--ent-coef", "0", "--out", str(BATCH)]
    source_files = [ROOT / p for p in ["finrl/experiments/train_scenario_mix_batch.py", "finrl/experiments/train_scenario_mix_logged.py", "finrl/experiments/scenario_mix.py", "finrl/experiments/scenario_mix_common.py", "finrl/experiments/notebook_utils.py", "finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py", "finrl/meta/env_portfolio_optimization/env_portfolio_optimization_gymnasium.py"]]
    launch = {
        "started_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "runtime_environment": settings, "launcher_pid": os.getpid(),
        "input_sha256": {str(p): sha(p) for p in [PRICES, SYNTHETIC]},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "paired_mixed_batch": str(MIXED), "selection_metric": "valid_sharpe",
        "test_evaluation_performed": False,
        "synthetic_argument_note": "The unchanged trainer requires an NPZ argument. It is loaded for CLI compatibility; p=0 disables all synthetic episode selection. All episode logs must contain only source=real, scenario=-1.",
        "runtime_note": "Same invocation-only QuantStats workaround as the paired mixed batch; PPO, reward, fees, architecture and evaluation code are unchanged.",
    }
    (runtime / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    (runtime / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    while not BATCH.exists() and process.poll() is None:
        time.sleep(.1)
    if BATCH.exists():
        archive = BATCH / "runtime"
        archive.mkdir(exist_ok=False)
        for name in ["launch.json", "preflight.json", "sitecustomize.py"]:
            shutil.copy2(runtime / name, archive / name)
    code = process.wait()
    if BATCH.exists():
        (BATCH / "runtime/launcher_exit.json").write_text(json.dumps({"returncode": code, "ended_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
