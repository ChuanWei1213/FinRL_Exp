"""Task-log display for the durable tmux session; exits with the experiment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
while True:
    status_file=a.out/'status.json'
    state=json.loads(status_file.read_text()) if status_file.exists() else {'status':'running','stage':'checkpoint_resume_verification'}
    jobs=[]
    for path in sorted((a.out/'jobs').glob('*/status.json')):
        d=json.loads(path.read_text());jobs.append((path.parent.name,d))
    finished=sum(d['status']=='complete' for _,d in jobs)
    print(datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S %Z'),
          'stage=',state.get('stage',state['status']),'completed jobs=',finished,'/',len(jobs),flush=True)
    for name,d in jobs:
        if d['status'] in ('running','failed'):
            print(' ',name,'update',d.get('update',0),'/',d.get('config',{}).get('updates','?'),
                  'steps',d.get('steps',0),'status',d['status'],flush=True)
    if state['status'] in ('complete','failed'):
        print(json.dumps(state,indent=2,ensure_ascii=False),flush=True)
        break
    time.sleep(30)
