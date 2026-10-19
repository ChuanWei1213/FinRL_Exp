"""Complete tables and figures after the authorized evaluation workflow finishes."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"results/scenario_mix/comprehensive_20261019"


def main():
    with (OUT/".tables_workflow.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/"evaluation_workflow_complete.json").exists():time.sleep(30)
        for name,args in [("summarize.py",["collect"]),("plot.py",[])]:
            subprocess.run([sys.executable,"-u","-B",str(Path(__file__).with_name(name)),*args],cwd=ROOT,check=True)
        print("Tables and figures ready for visual verification",flush=True)


if __name__=="__main__":main()
