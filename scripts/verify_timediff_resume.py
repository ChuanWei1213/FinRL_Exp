"""Exercise actual PPO updates and compare checkpoint-resume to continuous run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from finrl.experiments.timediff_scenario_pilot import write_json,parameter_hash,sha,BATCH


def main(out):
    out=out.resolve();base=out/'integration_smoke'
    if (base/'audit.json').exists():
        assert json.loads((base/'audit.json').read_text())['status']=='passed'
        return
    base.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name in ('resumed','continuous'):
        folder=base/name;folder.mkdir(exist_ok=True)
        config={'name':name,'case':'crash_2020_E0_f0','case_id':0,'group':'D','updates':2,'repeat':1,
                'pool_size':16,'banks':str(out/'banks'),'threads':2,'seed':0,'k_max':0,'commission':.0025,
                'source_code_sha256':sha(ROOT/'finrl/experiments/timediff_scenario_pilot.py')}
        path=folder/'job.json';write_json(path,config);paths.append(path)
    def execute(path,stop=None):
        command=[sys.executable,'-u','-m','finrl.experiments.timediff_scenario_pilot','worker','--job',str(path)]
        if stop is not None:command+=['--stop-after-update',str(stop)]
        subprocess.run(command,cwd=ROOT,check=True)
    execute(paths[0],1);execute(paths[0]);execute(paths[1])
    a=PPO.load(paths[0].parent/'latest_model.zip',device='cpu')
    b=PPO.load(paths[1].parent/'latest_model.zip',device='cpu')
    assert a.num_timesteps==b.num_timesteps==2*BATCH
    assert parameter_hash(a)==parameter_hash(b),'Resumed policy must match uninterrupted policy exactly'
    def tensors(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:tensors(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for p,q in zip(x,y):tensors(p,q)
        else:assert x==y
    tensors(a.policy.optimizer.state_dict(),b.policy.optimizer.state_dict())
    sa=json.loads((paths[0].parent/'resume.json').read_text());sb=json.loads((paths[1].parent/'resume.json').read_text())
    for name in ('controller','envs','torch_rng','numpy_rng','python_rng'):
        assert sa[name]==sb[name],name
    write_json(base/'audit.json',{'status':'passed','steps_per_branch':2*BATCH,'seed':0,
               'policy_parameters_exactly_identical':True,'optimizer_state_exactly_identical':True,
               'environment_and_rng_state_exactly_identical':True,'checkpoint_resume_after_update':1})
    print('PPO checkpoint/resume verification passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
