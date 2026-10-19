"""Wait for the authorized training, freeze validation choices, then evaluate test."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.scenario_mix.comprehensive_20261019 import evaluate as E


def call(*args):
    command=[sys.executable,"-u","-B",str(Path(__file__).with_name("evaluate.py")),*args]
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    with (E.OUT/".evaluation_workflow.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            status=json.loads((E.NEW/"status.json").read_text())
            failed=[k for k,v in status.items() if v["state"] in ["failed","cancelled"]]
            if failed:raise RuntimeError(f"Training failed: {failed}")
            if len(status)==30 and all(v["state"]=="succeeded" for v in status.values()):break
            time.sleep(30)
        print("All 30 new training jobs succeeded; evaluating train and validation",flush=True)
        call("pretest","--workers","5","--groups","REAL_REPLAY_PR30_B50","REAL_REPLAY_PR10_B50","FULL_B50")
        while True:
            expected=[E.OUT/"evaluations"/r["id"]/"pretest_complete.json" for r in E.records()]
            if all(p.exists() for p in expected):break
            time.sleep(30)
        call("freeze")
        print("Validation selections are frozen; starting authorized test diagnostics",flush=True)
        call("test","--workers","5")
        E.write(E.OUT/"evaluation_workflow_complete.json",{"completed_at":datetime.now(timezone.utc).isoformat(),"runs":90,"checkpoints":1800,"split_evaluations":5400})


if __name__=="__main__":main()
