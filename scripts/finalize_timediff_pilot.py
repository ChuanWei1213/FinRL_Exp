"""Wait for the tmux experiment, then independently audit and finalize artifacts."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out=a.out.resolve()
def write(value):
    temp=a.out/'finalization.json.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(a.out/'finalization.json')
write({'status':'waiting_for_steps1_3','updated_utc':datetime.now(timezone.utc).isoformat()})
while True:
    path=a.out/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {'status':'running'}
    if state['status']=='failed':
        write({'status':'failed','upstream':state});sys.exit(1)
    if state['status']=='complete':break
    time.sleep(30)
write({'status':'auditing','updated_utc':datetime.now(timezone.utc).isoformat()})
with (a.out/'final_audit.log').open('a') as log:
    completed=subprocess.run([sys.executable,str(Path(__file__).with_name('audit_timediff_pilot.py')),'--out',str(a.out)],stdout=log,stderr=subprocess.STDOUT)
write({'status':'complete' if completed.returncode==0 else 'failed','exit_code':completed.returncode,'report':str(a.out/'REPORT.md'),
       'updated_utc':datetime.now(timezone.utc).isoformat()})
sys.exit(completed.returncode)
