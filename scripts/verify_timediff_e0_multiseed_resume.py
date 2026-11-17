"""Actual nonzero-seed PPO resume and seed propagation checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from stable_baselines3 import PPO
from finrl.experiments import timediff_e0_multiseed as m


def main(out):
    os.sched_setaffinity(0,list(range(9))+list(range(12,21)))
    base=out/'integration_smoke';base.mkdir(exist_ok=True)
    if (base/'audit.json').exists():
        a=json.loads((base/'audit.json').read_text());assert a['status']=='passed' and a['module_sha256']==m.sha(m.__file__)
        return
    started=time.monotonic();paths={}
    for name,seed,updates in [('resumed',11,2),('continuous',11,2),('different_seed',12,1)]:
        folder=base/name;folder.mkdir(exist_ok=True)
        config={'name':name,'case':'crash_2020_E0_f0','case_id':0,'group':'D','updates':updates,
            'repeat':1,'pool_size':16,'banks':str(out/'banks'),'threads':2,'seed':seed,'phase':'development',
            'k_max':21,'commission':.0025,'backend':'BatchVecEnv','source_code_sha256':m.sha(m.__file__)}
        path=folder/'job.json';m.write_json(path,config);paths[name]=path
    def execute(name,stop=None):
        cmd=[sys.executable,'-u','-m','finrl.experiments.timediff_e0_multiseed','worker','--job',str(paths[name])]
        if stop is not None:cmd+=['--stop-after-update',str(stop)]
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    execute('resumed',1);execute('resumed');execute('continuous');execute('different_seed')
    torch.set_num_threads(2)
    a=PPO.load(paths['resumed'].parent/'latest_model.zip',device='cpu')
    b=PPO.load(paths['continuous'].parent/'latest_model.zip',device='cpu')
    c=PPO.load(paths['different_seed'].parent/'initial_model.zip',device='cpu')
    assert a.seed==b.seed==11 and c.seed==12
    assert m.parameter_hash(a)==m.parameter_hash(b)
    init=json.loads((paths['resumed'].parent/'initialization.json').read_text())
    assert init['seed']==11 and init['policy_sha256']!=m.parameter_hash(c)
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
    sa=json.loads((paths['resumed'].parent/'resume.json').read_text());sb=json.loads((paths['continuous'].parent/'resume.json').read_text())
    for key in ['controller','envs','last_episode_starts','torch_rng','numpy_rng','python_rng']:assert sa[key]==sb[key],key
    first=json.loads((paths['resumed'].parent/'updates.json').read_text())[0]
    assert first['unfinished_episodes_at_requested_switch']>0
    m.write_json(base/'audit.json',{'status':'passed','module_sha256':m.sha(m.__file__),
        'nonzero_seed_reaches_PPO_and_initial_weights':True,'different_seed_different_initialization':True,
        'policy_optimizer_env_rng_resume_bit_identical':True,'mid_episode_and_pool_transition_checked':True,
        'development_seeds':[11,12],'excluded_from_main_statistics':True,'elapsed_seconds':time.monotonic()-started})
    print((base/'audit.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
