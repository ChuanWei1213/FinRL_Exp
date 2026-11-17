"""TimeDiff lambda=.5: environment integration, paired PPO pilot, D calibration.

Only 2005-2017 training and 2018 validation episodes are exposed to policies.
No real event backtest, 2019 evaluation or formal multi-seed experiment is run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/cwei1213/TimeDiff/outputs/dow10/conditional/scenario_pilot/loss_ablation_20261115/lambda_0p5/generation')
TICKERS=['AAPL','AMGN','CRM','CSCO','IBM','INTC','MSFT','NKE','VZ','WMT']
EVENTS=['crash_2020','rebound_2020','bear_2022']
CASES=[f'{e}_{error}_f0' for e in EVENTS for error in ('E0','E1')]
N_ENVS,N_STEPS=16,504
BATCH=N_ENVS*N_STEPS
FEE=.0025


def now():return datetime.now(timezone.utc).isoformat()


def serial(x):
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(serial(payload),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def digest(payload):return hashlib.sha256(json.dumps(serial(payload),sort_keys=True).encode()).hexdigest()


def parameter_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def action_weights(action):
    logits=5*np.clip(np.asarray(action,dtype=np.float64),-1,1)
    exp=np.exp(logits-logits.max(axis=-1,keepdims=True))
    return exp/exp.sum(axis=-1,keepdims=True)


def transact(previous,target,gross,fee=FEE):
    """Self-financing fee: mu = 1-c*sum(abs(mu*w_stock-old_stock))."""
    previous,target,gross=map(lambda x:np.asarray(x,dtype=np.float64),(previous,target,gross))
    if not (np.isfinite(target).all() and (target>=0).all() and np.allclose(target.sum(axis=-1),1)):
        raise ValueError('Target must be finite long-only unit-sum weights')
    mu=np.ones(target.shape[:-1])
    for _ in range(30):
        updated=1-fee*np.abs(mu[...,None]*target[...,1:]-previous[...,1:]).sum(axis=-1)
        if np.max(np.abs(updated-mu))<1e-13:
            mu=updated;break
        mu=updated
    else:raise RuntimeError('Transaction cost did not converge')
    turnover=np.abs(mu[...,None]*target[...,1:]-previous[...,1:]).sum(axis=-1)
    weighted=target*gross
    growth=weighted.sum(axis=-1)
    factor=mu*growth
    final_weights=weighted/growth[...,None]
    if not ((factor>0).all() and np.isfinite(factor).all()):raise ValueError('Invalid portfolio wealth')
    return factor,final_weights,1-mu,turnover


def observations(prices,day,weights):
    """Causal HLC window, normalized only by its last known close."""
    cube=np.asarray(prices[:,day:day+50],dtype=np.float64).transpose(0,3,2,1)
    denominator=prices[:,day+49,:,0][:,None,:,None]
    return {'state':(cube/denominator).astype(np.float32),'last_action':np.asarray(weights,dtype=np.float32)}


def price_roundoff(values):
    """Snap only source floating-point HLC discrepancies below 1e-9 relative."""
    assert np.isfinite(values).all() and (values>0).all()
    close=values[...,0]
    bad_high=values[...,1]<close;bad_low=values[...,2]>close
    assert (values[...,1]>=close*(1-1e-9)).all()
    assert (values[...,2]<=close*(1+1e-9)).all()
    counts={'high_roundoff_snaps':int(bad_high.sum()),'low_roundoff_snaps':int(bad_low.sum()),'relative_tolerance':1e-9}
    values[...,1]=np.maximum(values[...,1],close)
    values[...,2]=np.minimum(values[...,2],close)
    return counts


def prepare(out,source=SOURCE):
    banks=out/'banks'
    if (banks/'manifest.json').exists():return json.loads((banks/'manifest.json').read_text())
    banks.mkdir(parents=True,exist_ok=True)
    m=json.loads((source/'manifest.json').read_text())
    assert m['generator_metadata']['args']['lambda_condition']==.5
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    assert sha(m['data'])==m['data_sha256']
    frame=pd.read_csv(m['data'],parse_dates=['date'])
    # Later real prices are excluded before any bank is built.
    frame=frame[frame.date<='2018-12-31']
    wide={k:frame.pivot(index='date',columns='tic',values=k)[TICKERS] for k in ('close','high','low')}
    dates=wide['close'].index
    prices=np.stack([wide[k].to_numpy() for k in ('close','high','low')],axis=-1)
    result={'source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),'checkpoint_sha256':m['checkpoint_sha256'],
            'data_sha256':m['data_sha256'],'observation_keys':['state','last_action'],'obs_days':50,'episode_days':21,'k_max':0,
            'commission':FEE,'reward':'100*log(net_wealth_t/net_wealth_previous)','initial_weights':'all cash; first purchase charged',
            'normalization':'each observed HLC window divided by that asset last known close; no fitted RL scaler',
            'source_note':'only train/validation paths read; no 2019 paths or real event evaluation','banks':{}}
    for split,start,end,stride in [('train','2005-01-01','2017-12-31',1),('validation','2018-01-01','2018-12-31',21)]:
        eligible=np.flatnonzero((dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end)))
        starts=np.arange(eligible[0]+50,eligible[-1]-20+1,stride)
        values=np.stack([prices[s-50:s+21] for s in starts])
        rounding=price_roundoff(values)
        name=f'real_{split}'
        np.save(banks/f'{name}.npy',values)
        result['banks'][name]={'n':len(values),'sha256':sha(banks/f'{name}.npy'),'history_start':str(dates[starts[0]-50].date()),
                               'first_target':str(dates[starts[0]].date()),'last_target':str(dates[starts[-1]+20].date()),'target_stride':stride,**rounding}
    for fid in CASES+['H_historical']:
        for split in ('train','validation'):
            path=source/fid/split/'paths.npz'
            summary=json.loads((source/fid/split/'summary.json').read_text())
            assert summary['status']=='complete' and sha(path)==summary['artifacts_sha256']['paths.npz']
            with np.load(path,allow_pickle=False) as z:
                assert z['tickers'].tolist()==TICKERS
                index=z['history_index']
                history=np.stack([z[f'history_{k}'][index,-50:] for k in ('close','high','low')],axis=-1)
                future=np.stack([z[k] for k in ('close','high','low')],axis=-1)
                np.testing.assert_allclose(history[:,-1,:,0],z['anchor_close'],rtol=0,atol=0)
                values=np.concatenate([history,future],axis=1)
                rounding=price_roundoff(values)
            name=f'{fid}_{split}'
            np.save(banks/f'{name}.npy',values)
            result['banks'][name]={'n':len(values),'sha256':sha(banks/f'{name}.npy'),'source_paths_sha256':sha(path),
                                   'source':str(path),'labels_in_observation':False,**rounding}
    write_json(banks/'manifest.json',result)
    write_json(out/'step1_data_adapter.json',{'status':'passed','real_validation_episodes':result['banks']['real_validation']['n'],
               'synthetic_training_per_condition':800,'synthetic_validation_per_condition':200,'all_paths_have_50_history_and_21_future':True,
               'generated_target_oracle_labels_excluded_from_observation':True,'manifest':str(banks/'manifest.json')})
    return result


class Controller:
    def __init__(self,real,synthetic,mode,case_id=0,pool_size=16):
        self.real,self.synthetic,self.mode=real,synthetic,mode
        self.pool_size=pool_size
        self.case_id=case_id
        self.rng=np.random.default_rng(np.random.SeedSequence([0,100,case_id]))
        self.samplers=[np.random.default_rng(np.random.SeedSequence([0,200,case_id,i])) for i in range(16)]
        self.generation=0
        self.members=[]
        self.rotate()

    def rotate(self):
        if self.mode=='D':
            half=self.pool_size//2
            self.members=[(0,int(i)) for i in self.rng.choice(len(self.real),half,replace=False)]
            self.members += [(1,int(i)) for i in self.rng.choice(len(self.synthetic),half,replace=False)]
        self.generation+=1

    def pick(self,slot):
        source=0 if self.mode=='R' or slot<8 else 1
        if self.mode=='D':
            # Eight active environments per source; larger pools broaden the
            # retained paths without changing PPO batch size or parallelism.
            half=self.pool_size//2
            members=self.members[:half] if source==0 else self.members[half:]
            assigned=members[slot%8::8]
            source,index=assigned[int(self.samplers[slot].integers(len(assigned)))]
        else:
            bank=self.real if source==0 else self.synthetic
            index=int(self.samplers[slot].integers(len(bank)))
        return source,index

    def prices(self,source,index):return (self.real if source==0 else self.synthetic)[index]

    def replay(self,envs):
        members=self.members if self.mode=='D' else [(e.source,e.index) for e in envs]
        return np.stack([self.prices(source,index) for source,index in members])

    def state(self):
        return {'rng':self.rng.bit_generator.state,'samplers':[s.bit_generator.state for s in self.samplers],
                'members':self.members,'generation':self.generation}

    def restore(self,state):
        self.rng.bit_generator.state=state['rng']
        for rng,value in zip(self.samplers,state['samplers']):rng.bit_generator.state=value
        self.members=[tuple(v) for v in state['members']];self.generation=state['generation']


class PilotEnv(gym.Env):
    metadata={}

    def __init__(self,controller,slot=0):
        self.controller,self.slot=controller,slot
        self.observation_space=spaces.Dict({'state':spaces.Box(0,np.inf,(3,10,50),np.float32),
                                            'last_action':spaces.Box(0,1,(11,),np.float32)})
        self.action_space=spaces.Box(-1,1,(11,),np.float32)
        self.source,self.index=0,0
        self.day=0;self.wealth=1.;self.weights=np.r_[1.,np.zeros(10)]
        self.completed=np.zeros(2,dtype=int)

    def _obs(self):
        obs=observations(self.controller.prices(self.source,self.index)[None],self.day,self.weights[None])
        return {key:value[0] for key,value in obs.items()}

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.controller.samplers[self.slot]=np.random.default_rng(np.random.SeedSequence([seed,200,self.controller.case_id,self.slot]))
        self.source,self.index=self.controller.pick(self.slot)
        self.day,self.wealth=0,1.
        self.weights=np.r_[1.,np.zeros(10)]
        return self._obs(),{}

    def step_weights(self,target):
        if self.day>=21:raise RuntimeError('Episode already complete; no dummy terminal reward')
        prices=self.controller.prices(self.source,self.index)
        gross=np.r_[1.,prices[self.day+50,:,0]/prices[self.day+49,:,0]]
        factor,weights,cost,turnover=transact(self.weights,target,gross)
        previous=self.wealth
        exposure_delta=float(np.sum(target[1:])-np.sum(self.weights[1:]))
        self.wealth*=float(factor);self.weights=weights;self.day+=1
        reward=100*math.log(float(factor))
        done=self.day==21
        info={'wealth':self.wealth,'previous_wealth':previous,'cost_fraction':float(cost),'turnover':float(turnover),
              'exposure_change_from_trade':exposure_delta,'source':self.source,'path_id':self.index}
        if done:
            self.completed[self.source]+=1
            info['episode']={'r':100*math.log(self.wealth),'l':21}
        return self._obs(),reward,done,False,info

    def step(self,action):return self.step_weights(action_weights(action))

    def state(self):return {'source':self.source,'index':self.index,'day':self.day,'wealth':self.wealth,'weights':self.weights,'completed':self.completed}

    def restore(self,s):
        for key in ('source','index','day','wealth'):setattr(self,key,s[key])
        self.weights=np.array(s['weights']);self.completed=np.array(s['completed'])


class EIIEWindowExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=211)
        self.conv=nn.Sequential(nn.Conv2d(3,2,(1,3)),nn.ReLU(),nn.Conv2d(2,20,(1,48)),nn.ReLU(),nn.Flatten())

    def forward(self,observations):return torch.cat([self.conv(observations['state']),observations['last_action']],dim=1)


def evaluate(model,prices):
    prices=np.asarray(prices)
    wealth=np.ones(len(prices));weights=np.zeros((len(prices),11));weights[:,0]=1
    with torch.inference_mode():
        for day in range(21):
            action,_=model.predict(observations(prices,day,weights),deterministic=True)
            target=action_weights(action)
            gross=np.column_stack([np.ones(len(prices)),prices[:,day+50,:,0]/prices[:,day+49,:,0]])
            factor,weights,_cost,_turnover=transact(weights,target,gross)
            wealth*=factor
    return 100*np.log(wealth)


def se(values):
    values=np.asarray(values)
    return float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.


def worker(job_path,stop_after=None):
    job_path=Path(job_path);config=json.loads(job_path.read_text());out=job_path.parent
    torch.set_num_threads(config.get('threads',2))
    torch.set_num_interop_threads(1)
    banks=Path(config['banks'])
    real=np.load(banks/'real_train.npy',mmap_mode='r');val=np.load(banks/'real_validation.npy',mmap_mode='r')
    synthetic=None;synval=None
    if config['group']!='R':
        fid='H_historical' if config['group']=='H' else config['case']
        synthetic=np.load(banks/f'{fid}_train.npy',mmap_mode='r')
        synval=np.load(banks/f'{fid}_validation.npy',mmap_mode='r')
    controller=Controller(real,synthetic,config['group'],config['case_id'],config.get('pool_size',16))
    envs=[PilotEnv(controller,i) for i in range(16)]
    vec=DummyVecEnv([lambda e=e:e for e in envs])
    checkpoint=out/'latest_model.zip';resume=out/'resume.json'
    records=[];evaluations=[]
    config_hash=digest(config)
    statepath=out/'status.json'
    if statepath.exists():
        prior=json.loads(statepath.read_text())
        assert prior['config_sha256']==config_hash
        if prior['status']=='complete':return
    write_json(statepath,{'status':'running','started_utc':now(),'config_sha256':config_hash,'config':config})
    if resume.exists() and checkpoint.exists():
        s=json.loads(resume.read_text())
        controller.restore(s['controller'])
        for env,value in zip(envs,s['envs']):env.restore(value)
        model=PPO.load(out/s['checkpoint'],env=vec,device='cpu',force_reset=False)
        model._last_obs={key:np.stack([e._obs()[key] for e in envs]) for key in ('state','last_action')}
        model._last_episode_starts=np.ones(16,dtype=bool)
        vec._seeds=[None]*16
        torch.set_rng_state(torch.tensor(s['torch_rng'],dtype=torch.uint8))
        ns=s['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        random.setstate((s['python_rng'][0],tuple(s['python_rng'][1]),s['python_rng'][2]))
        records=[r for r in json.loads((out/'updates.json').read_text()) if r['update']<=s['update']]
        evaluations=[r for r in json.loads((out/'validation_returns.json').read_text()) if r['update']<=s['update']]
        initial=json.loads((out/'initialization.json').read_text())
        best=s['best'];age=s['pool_age'];plateau=s['plateau']
        before=np.array(evaluations[-1]['real'])
        start_update=s['update']+1
    else:
        model=PPO('MultiInputPolicy',vec,seed=0,device='cpu',n_steps=N_STEPS,batch_size=504,n_epochs=10,
                  learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,verbose=0,
                  policy_kwargs={'features_extractor_class':EIIEWindowExtractor,'net_arch':dict(pi=[64],vf=[64])})
        initial={'policy_sha256':parameter_hash(model),'seed':0,'parameter_count':sum(p.numel() for p in model.policy.parameters())}
        write_json(out/'initialization.json',initial)
        before=evaluate(model,val)
        evaluations=[{'update':0,'real':before.tolist(),'synthetic':evaluate(model,synval).tolist() if synval is not None else []}]
        best=float(before.mean());age=0;plateau=0;start_update=1
        model.save(out/'initial_model')
        model.save(out/'best_model')
    model.set_logger(configure(str(out/'ppo'),['csv']))
    started=time.monotonic()
    final_update=min(config['updates'],stop_after) if stop_after is not None else config['updates']
    for update in range(start_update,final_update+1):
        replay=controller.replay(envs) if config['group']=='D' else None
        replay_before=evaluate(model,replay) if replay is not None else None
        tick=time.monotonic()
        model.learn(total_timesteps=BATCH,reset_num_timesteps=False,progress_bar=False,log_interval=1)
        training_seconds=time.monotonic()-tick
        assert model.num_timesteps==update*BATCH
        assert all(env.day==0 for env in envs),'Every PPO update must end on an episode boundary'
        tick=time.monotonic()
        after=evaluate(model,val)
        synthetic_returns=evaluate(model,synval) if synval is not None else np.array([])
        replay_delta=evaluate(model,replay)-replay_before if replay is not None else np.array([])
        evaluation_seconds=time.monotonic()-tick
        change=after-before
        age+=1;reason='continue';rotate=False
        if config['group']=='D':
            if config.get('adaptive'):
                thresholds=config['adaptive']
                plateau=plateau+1 if change.mean()<=thresholds['improvement_threshold'] else 0
                if age>=thresholds['minimum_repeats'] and change.mean() < -thresholds['regression_threshold']:
                    reason='validation_regression';rotate=True
                elif age>=thresholds['minimum_repeats'] and plateau>=2:
                    reason='plateau';rotate=True
                elif age>=thresholds['maximum_repeats']:
                    reason='maximum_repeats';rotate=True
            elif age>=config['repeat']:
                reason='fixed_repeat_limit';rotate=True
        record={'update':update,'steps':model.num_timesteps,'real_validation_score':float(after.mean()),
                'synthetic_validation_score':float(synthetic_returns.mean()) if len(synthetic_returns) else None,
                'validation_gain':float(change.mean()),'validation_gain_se':se(change),
                'replay_gain':float(replay_delta.mean()) if len(replay_delta) else None,'replay_gain_se':se(replay_delta) if len(replay_delta) else None,
                'pool_generation':controller.generation,'pool_age':age,'switch_reason':reason,
                'training_seconds':training_seconds,'validation_seconds':evaluation_seconds,
                'completed_real_episodes':int(sum(e.completed[0] for e in envs)),
                'completed_synthetic_episodes':int(sum(e.completed[1] for e in envs)),
                'approx_kl':float(model.logger.name_to_value.get('train/approx_kl',0))}
        if config['group']!='R':assert record['completed_real_episodes']==record['completed_synthetic_episodes']
        records.append(record);evaluations.append({'update':update,'real':after.tolist(),'synthetic':synthetic_returns.tolist()})
        if after.mean()>best:
            best=float(after.mean());model.save(out/'best_model')
        if rotate:
            controller.rotate();age=0;plateau=0
            model._last_obs=vec.reset();model._last_episode_starts=np.ones(16,dtype=bool)
        tick=time.monotonic()
        versioned=out/f'checkpoint_{update:03d}.zip'
        model.save(versioned)
        shutil.copy2(versioned,checkpoint)
        record['checkpoint_seconds']=time.monotonic()-tick
        numpy_rng=np.random.get_state()
        write_json(out/'updates.json',records);write_json(out/'validation_returns.json',evaluations)
        pd.DataFrame(records).to_csv(out/'updates.csv',index=False)
        write_json(resume,{'update':update,'checkpoint':versioned.name,'controller':controller.state(),'envs':[e.state() for e in envs],
                   'torch_rng':torch.get_rng_state().tolist(),'numpy_rng':numpy_rng,'python_rng':random.getstate(),
                   'best':best,'pool_age':age,'plateau':plateau})
        before=after
        write_json(statepath,{'status':'running','config_sha256':config_hash,'config':config,'update':update,
                   'steps':model.num_timesteps,'initialization':initial,'best_validation_score':best,'updated_utc':now()})
        print(f"{config['name']} update={update}/{config['updates']} steps={model.num_timesteps} validation={after.mean():.6f} switch={reason}",flush=True)
    if model.num_timesteps<config['updates']*BATCH:
        write_json(statepath,{'status':'checkpoint_ready','config_sha256':config_hash,'config':config,'steps':model.num_timesteps,'updated_utc':now()})
        vec.close();return
    write_json(statepath,{'status':'complete','config_sha256':config_hash,'config':config,'steps':model.num_timesteps,
               'best_validation_score':best,'last_validation_score':float(before.mean()),'initialization':initial,
               'wall_seconds_this_process':time.monotonic()-started,'completed_utc':now()})
    vec.close()


def make_job(out,name,case,group,updates,repeat=1,pool=16,adaptive=None):
    folder=out/'jobs'/name;folder.mkdir(parents=True,exist_ok=True)
    config={'name':name,'case':case,'case_id':EVENTS.index(case.rsplit('_',2)[0]) if case in CASES else EVENTS.index(case),
            'group':group,'updates':updates,'repeat':repeat,'pool_size':pool,'banks':str(out/'banks'),'threads':2,
            'seed':0,'k_max':0,'commission':FEE,'source_code_sha256':sha(__file__)}
    if adaptive is not None:config['adaptive']=adaptive
    target=folder/'job.json'
    if target.exists():assert json.loads(target.read_text())==config
    else:write_json(target,config)
    return target


def run_jobs(out,jobs,workers,stage):
    write_json(out/'status.json',{'status':'running','stage':stage,'jobs':[str(j.parent) for j in jobs],'updated_utc':now()})
    def one(path):
        state=path.parent/'status.json'
        if state.exists() and json.loads(state.read_text())['status']=='complete':return str(path.parent),'cached'
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (path.parent/'training.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u','-m','finrl.experiments.timediff_scenario_pilot','worker','--job',str(path)],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Job failed: {path.parent}; see training.log')
        return str(path.parent),'complete'
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(one,j):j for j in jobs}
        for future in as_completed(futures):
            path,state=future.result();print(stage,state,path,flush=True)


def read_scores(path,updates=8):
    evaluations=json.loads((path.parent/'validation_returns.json').read_text())
    selected=[row['real'] for row in evaluations if max(1,updates-3)<=row['update']<=updates]
    return np.mean(selected,axis=0)


def calibration(out,workers):
    grid={}
    jobs=[]
    for case in CASES:
        for repeat in (1,2,4,8):
            path=make_job(out,f'{case}_D_repeat{repeat}_pool16',case,'D',8,repeat)
            grid[(case,repeat,16)]=path;jobs.append(path)
    run_jobs(out,jobs,workers,'step3_fixed_repeat_calibration')
    cpaths={case:out/'jobs'/f'{case}_C'/'job.json' for case in CASES}
    comparisons=[];errors=[];overfit=False;plateau_ages=[]
    for (case,repeat,pool),path in grid.items():
        diff=read_scores(path)-read_scores(cpaths[case])
        logs=json.loads((path.parent/'updates.json').read_text())
        for r in logs:
            errors.append(r['validation_gain_se'])
            if r['replay_gain']>r['replay_gain_se'] and r['validation_gain'] < -r['validation_gain_se']:overfit=True
        comparisons.append({'case':case,'repeat':repeat,'pool_size':pool,'paired_validation_gain_vs_C':float(diff.mean()),
                            'paired_se':se(diff),'episode_differences':diff.tolist()})
    by_repeat={k:np.mean([r['episode_differences'] for r in comparisons if r['repeat']==k],axis=0) for k in (1,2,4,8)}
    best_repeat=max(by_repeat,key=lambda k:float(by_repeat[k].mean()))
    if overfit:
        larger=[]
        for case in CASES:
            for pool in (32,64):
                path=make_job(out,f'{case}_D_repeat{best_repeat}_pool{pool}',case,'D',8,best_repeat,pool)
                grid[(case,best_repeat,pool)]=path;larger.append(path)
        run_jobs(out,larger,workers,'step3_larger_pool_check')
        for (case,repeat,pool),path in grid.items():
            if pool==16:continue
            diff=read_scores(path)-read_scores(cpaths[case])
            comparisons.append({'case':case,'repeat':repeat,'pool_size':pool,'paired_validation_gain_vs_C':float(diff.mean()),
                                'paired_se':se(diff),'episode_differences':diff.tolist()})
    pool_vectors={}
    for pool in (16,32,64):
        rows=[r['episode_differences'] for r in comparisons if r['repeat']==best_repeat and r['pool_size']==pool]
        if rows:pool_vectors[pool]=np.mean(rows,axis=0)
    best_pool=max(pool_vectors,key=lambda p:float(pool_vectors[p].mean()))
    # One-standard-error tie rule favors the smaller pool. Episode differences
    # are averaged over cases first; repeated case labels do not multiply n.
    chosen_pool=min(p for p,v in pool_vectors.items() if np.mean(pool_vectors[best_pool]-v)<=se(pool_vectors[best_pool]-v)+1e-12)
    minimum=min(k for k,v in by_repeat.items() if k<=best_repeat and np.mean(by_repeat[best_repeat]-v)<=se(by_repeat[best_repeat]-v)+1e-12)
    valid_errors=np.array(errors,dtype=float)
    settings={'environment_pool_size':chosen_pool,'active_parallel_environments':16,'minimum_repeats':minimum,'maximum_repeats':best_repeat,
              'improvement_threshold':max(float(np.median(valid_errors)),1e-10),
              'regression_threshold':max(float(np.quantile(valid_errors,.9)),1e-10),'plateau_patience':2,
              'threshold_units':'sum of 100*log(net wealth ratios) over a 21-day episode',
              'threshold_estimation':'median / 90th percentile of paired real-2018 validation update standard errors',
              'best_fixed_repeat':best_repeat,'best_raw_pool':best_pool,'larger_pool_triggered':overfit,
              'calibration_interactions_per_candidate':8*BATCH,
              'selection_score':'mean paired real-2018 validation gain over updates 5-8 versus C at the same step',
              'pooled_case_score':float(pool_vectors[chosen_pool].mean()),'formal_parameters_locked':False,
              'limitation':'single development seed, few non-overlapping real-2018 episodes; no event performance or 2019 evaluation'}
    pd.DataFrame([{k:v for k,v in r.items() if k!='episode_differences'} for r in comparisons]).to_csv(out/'D_calibration_candidates.csv',index=False)
    write_json(out/'D_calibration_evidence.json',comparisons)
    write_json(out/'D_selected_parameters.json',settings)
    adaptive={k:settings[k] for k in ('minimum_repeats','maximum_repeats','improvement_threshold','regression_threshold')}
    final=[make_job(out,f'{case}_D_adaptive',case,'D',32,pool=chosen_pool,adaptive=adaptive) for case in CASES]
    run_jobs(out,final,workers,'step3_adaptive_D_validation')
    return settings


def summarize(out,settings):
    rows=[];initials=set()
    for path in sorted((out/'jobs').glob('*/status.json')):
        s=json.loads(path.read_text());assert s['status']=='complete'
        c=s['config'];initials.add(s['initialization']['policy_sha256'])
        logs=pd.read_csv(path.parent/'updates.csv')
        rows.append({'name':c['name'],'group':c['group'],'steps':s['steps'],'best_2018_real_validation':s['best_validation_score'],
                     'last_2018_real_validation':s['last_validation_score'],'training_seconds':logs.training_seconds.sum(),
                     'validation_seconds':logs.validation_seconds.sum(),'real_episodes':int(logs.completed_real_episodes.iloc[-1]),
                     'synthetic_episodes':int(logs.completed_synthetic_episodes.iloc[-1])})
    assert len(initials)==1,'Paired models must start from identical parameters'
    pd.DataFrame(rows).to_csv(out/'job_summary.csv',index=False)
    matrix=[]
    for case in CASES:
        event=case.rsplit('_',2)[0]
        for group in ('R','H','C','D'):
            name=f'{event}_{group}' if group in ('R','H') else f'{case}_{group}' if group=='C' else f'{case}_D_adaptive'
            matrix.append({'case':case,'group':group,'checkpoint':str(out/'jobs'/name/'best_model.zip'),
                           'last_checkpoint':str(out/'jobs'/name/'latest_model.zip')})
    pd.DataFrame(matrix).to_csv(out/'paired_group_checkpoints.csv',index=False)
    report=['# TimeDiff λ=0.5：步驟 1–3 前測及 D 校準','',
            '本輪僅使用 2005–2017 訓練與 2018 驗證；未讀取 2019 合成測試路徑、未執行 2020／2022 真實事件回測。',
            'seed=0 僅作開發前測。本輪使用 CPU、PPO EIIE 特徵、16 個平行環境；每環境 rollout=504、每更新 8,064 次互動。',
            'R／H／C／最後選定的 D 各跑 32 更新=258,048 次互動；D 固定重複候選各跑 8 更新=64,512 次互動。此為完整 21 日 episode 對齊，原暫定正式步數仍待下一階段確認。','',
            '## 步驟 1：資料及環境','',
            '觀測僅含 50 日 HLC 與前次實際配置；每條訓練 episode 21 步，不添加額外終止 reward。初始全現金，首次買入照常計費。',
            '費用以自融資方程 mu=1-c*sum(abs(mu*target_stock-previous_stock)) 求解；reward=100*log(扣費後資產比)。股票及現金配置非負且合計為一。',
            'RL 使用當前觀測窗最後可知 close 作相對價格標準化；不額外擬合使用未來資料的 scaler。無預測目標、事件標籤或來源旗標進入 observation。','',
            '## 步驟 2：R／H／C／D','',
            '同一 seed 的初始策略參數雜湊全部一致。每個事件獨立訓練；R/H 在同事件 E0/E1 的訓練資料及初次時程相同，引用同事件的相同 checkpoint，不跨事件移轉。',
            'H/C/D 使用八個真實與八個合成活躍環境，每個 episode 等長，因此已完成的 episode 與互動比例均精確為 50%。C/D 使用相同合成路徑池與總互動數。','',
            '## 步驟 3：D 參數及選擇依據','',
            '先在 16 條環境路徑池比較固定重複 1/2/4/8 個完整更新週期。每次 PPO 更新都重新收集 rollout，不重用舊 rollout。',
            '校準主指標是相對同步 C 策略，在固定、不重疊的真實 2018 驗證 episode 上更新 5–8 的平均配對改善。合成 2018 驗證、重播池改善與未改善紀錄另外保存。',
            '若重播改善大於其標準誤、但真實驗證退步大於其標準誤，對較佳重複週期再比較 32/64 條路徑池；活躍環境仍為 16，保持 PPO 更新批次固定。',
            '池大小與最短更新數採配對一個標準誤規則偏向較小值；改善及退步門檻取驗證更新配對標準誤的中位數及第 90 百分位。','',
            '```json',json.dumps(serial(settings),indent=2,ensure_ascii=False),'```','',
            '這些是本輪開發校準數值；正式設定尚需依原計畫步驟 4 的耗時、收斂與正式種子規劃鎖定。沒有強制把 D 結果解讀為優於 C。',
            'adaptive D 遇到平台（連續兩次未超過改善門檻）、驗證退化或最長次數即切換路徑池，且僅在完整 episode／PPO 更新邊界切換。原因及所有未改善更新均記錄於 updates.csv。','',
            '## 交付','',f'- [D 候選比較]({out/"D_calibration_candidates.csv"})。',f'- [選定參數]({out/"D_selected_parameters.json"})。',
            f'- [所有工作摘要]({out/"job_summary.csv"})。',f'- [24 組配對 checkpoint 索引]({out/"paired_group_checkpoints.csv"})。',
            f'- [資料接入稽核]({out/"step1_data_adapter.json"})。',
            '每個工作保存 initial/best/latest checkpoint、optimizer、每次更新的驗證向量、pool 排程、RNG 與環境續跑狀態。','']
    (out/'REPORT.md').write_text('\n'.join(report))
    write_json(out/'status.json',{'status':'complete','scope':'steps1-3 only; seed=0; no real event scoring','jobs':len(rows),
               'identical_initial_policy_sha256':next(iter(initials)),'completed_utc':now(),'report':str(out/'REPORT.md')})


def run(out,workers=2,resume=False):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()) and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    try:
        write_json(out/'plan.json',{'created_utc':now(),'scope':'steps1-3 only','source':str(SOURCE),'seed':0,
                  'initial_updates':32,'calibration_updates':8,'n_envs':16,'n_steps':504,'batch_size':504,'n_epochs':10,
                  'workers':workers,'device':'cpu','repeats':[1,2,4,8],'pool_sizes_if_overfit':[16,32,64],
                  'validation_only':'real and synthetic 2018; no 2019 or event scores','code_sha256':sha(__file__)})
        write_json(out/'status.json',{'status':'running','stage':'step1_data_adapter','updated_utc':now()})
        prepare(out)
        initial=[]
        for event in EVENTS:
            for group in ('R','H'):initial.append(make_job(out,f'{event}_{group}',event,group,32))
        for case in CASES:initial.append(make_job(out,f'{case}_C',case,'C',32))
        run_jobs(out,initial,workers,'step2_R_H_C_training')
        settings=calibration(out,workers)
        summarize(out,settings)
    except Exception as exc:
        write_json(out/'status.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','updated_utc':now(),
                   'resume_command':f'{sys.executable} -m finrl.experiments.timediff_scenario_pilot run --out {out} --workers {workers} --resume'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    main=sub.add_parser('run');main.add_argument('--out',type=Path,required=True);main.add_argument('--workers',type=int,default=2);main.add_argument('--resume',action='store_true')
    w=sub.add_parser('worker');w.add_argument('--job',type=Path,required=True);w.add_argument('--stop-after-update',type=int)
    prep=sub.add_parser('prepare');prep.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.cmd=='worker':worker(a.job,a.stop_after_update)
    elif a.cmd=='prepare':prepare(a.out)
    else:run(a.out,a.workers,a.resume)
