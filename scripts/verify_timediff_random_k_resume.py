"""Check actual PPO resume while episodes cross rollouts and D requests a pool switch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from stable_baselines3 import PPO
from finrl.experiments import timediff_random_k as r


def main(out):
    base=out/'integration_smoke';base.mkdir(exist_ok=True)
    audit=base/'audit.json'
    if audit.exists():
        assert json.loads(audit.read_text())['status']=='passed'
        return
    jobs=[];started=time.monotonic()
    for name in ('resumed','continuous'):
        folder=base/name;folder.mkdir(exist_ok=True)
        config={'name':name,'case':'crash_2020_E0_f0','case_id':0,'group':'D','updates':2,
            'repeat':1,'pool_size':16,'banks':str(out/'banks'),'threads':2,'seed':0,
            'k_max':21,'commission':.0025,'source_code_sha256':r.sha(r.__file__)}
        path=folder/'job.json';r.write_json(path,config);jobs.append(path)
    def execute(path,stop=None):
        cmd=[sys.executable,'-u','-m','finrl.experiments.timediff_random_k','worker','--job',str(path)]
        if stop is not None:cmd+=['--stop-after-update',str(stop)]
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    execute(jobs[0],1);execute(jobs[0]);execute(jobs[1])
    torch.set_num_threads(2)
    a=PPO.load(jobs[0].parent/'latest_model.zip',device='cpu')
    b=PPO.load(jobs[1].parent/'latest_model.zip',device='cpu')
    assert a.num_timesteps==b.num_timesteps==2*r.BATCH
    assert r.parameter_hash(a)==r.parameter_hash(b)
    def compare(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:compare(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y):compare(u,v)
        else:assert x==y
    compare(a.policy.optimizer.state_dict(),b.policy.optimizer.state_dict())
    sa=json.loads((jobs[0].parent/'resume.json').read_text());sb=json.loads((jobs[1].parent/'resume.json').read_text())
    for key in ('controller','envs','last_episode_starts','torch_rng','numpy_rng','python_rng'):assert sa[key]==sb[key],key
    first=json.loads((jobs[0].parent/'updates.json').read_text())[0]
    assert first['unfinished_episodes_at_requested_switch']>0
    assert any(e['day']>0 for e in sa['envs'])
    rows=json.loads((jobs[1].parent/'updates.json').read_text())
    r.write_json(audit,{'status':'passed','checkpoint_mid_episode':True,'natural_pool_transition':True,
        'policy_optimizer_environment_rng_bit_identical':True,'steps_per_branch':2*r.BATCH,
        'elapsed_seconds':time.monotonic()-started,'training_seconds_per_update':sum(x['training_seconds'] for x in rows)/2,
        'validation_seconds_per_update':sum(x['validation_seconds'] for x in rows)/2,
        'module_sha256':r.sha(r.__file__)})
    print(audit.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
