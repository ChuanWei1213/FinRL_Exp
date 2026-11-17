"""Seed-0 PPO rerun with a real prelude k ~ Uniform{0,...,21}.

Frozen TimeDiff lambda=.5 paths. Real train<=2017, selection on 2018 only.
Every training transition is budgeted; partial episodes continue across PPO
updates and finish naturally before a requested D pool change takes effect.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from finrl.experiments import timediff_scenario_pilot as original

ROOT, SOURCE = original.ROOT, original.SOURCE
CASES, EVENTS, TICKERS = original.CASES, original.EVENTS, original.TICKERS
N_ENVS, N_STEPS, BATCH, FEE = 16, 504, 8064, .0025
K_MAX, HISTORY, HORIZON = 21, 71, 21
now, sha, write_json, digest, se = original.now, original.sha, original.write_json, original.digest, original.se
action_weights, transact, observations = original.action_weights, original.transact, original.observations
EIIEWindowExtractor, parameter_hash = original.EIIEWindowExtractor, original.parameter_hash


def build_banks(out, test=False):
    """All 22 possible k values fit before choosing a path; never redraw k."""
    out = Path(out)
    banks = out / 'banks'
    banks.mkdir(parents=True, exist_ok=True)
    target = banks / ('test_manifest.json' if test else 'manifest.json')
    if target.exists():
        return json.loads(target.read_text())
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    assert manifest['generator_metadata']['args']['lambda_condition'] == .5
    assert sha(manifest['data']) == manifest['data_sha256']
    assert sha(manifest['checkpoint']) == manifest['checkpoint_sha256']
    frame = pd.read_csv(manifest['data'], parse_dates=['date'])
    # Filter before pivoting; event prices never enter PPO banks/evaluation.
    frame = frame[frame.date <= ('2019-12-31' if test else '2018-12-31')]
    wide = {k:frame.pivot(index='date', columns='tic', values=k)[TICKERS] for k in ('close','high','low')}
    dates = wide['close'].index
    prices = np.stack([wide[k].to_numpy() for k in ('close','high','low')],axis=-1)
    result = {'created_utc':now(), 'k_max':K_MAX, 'source':str(SOURCE),
        'generator_checkpoint_sha256':manifest['checkpoint_sha256'], 'price_source_sha256':manifest['data_sha256'],
        'history_rows':HISTORY, 'future_rows':21, 'obs_days':50,
        'layout':'50 observed + k real traded + 21 real/synthetic traded; k redrawn each reset',
        'path_eligibility':'entire 71-day history and 21-day target inside its split for every k',
        'normalization':'last known close; no new fitted scaler', 'banks':{}}
    splits = [('test','2019-01-01','2019-12-31',21)] if test else [
        ('train','2005-01-01','2017-12-31',1),('validation','2018-01-01','2018-12-31',21)]
    for split,start,end,stride in splits:
        eligible = np.flatnonzero((dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end)))
        lo,hi = int(eligible[0]),int(eligible[-1])
        starts = np.arange(lo+HISTORY, hi-HORIZON+2, stride)
        values = np.stack([prices[s-HISTORY:s+HORIZON] for s in starts])
        rounding = original.price_roundoff(values)
        name = f'real_{split}'
        np.save(banks/f'{name}.npy',values)
        result['banks'][name] = {'n':len(values),'sha256':sha(banks/f'{name}.npy'),
            'split_start':start,'split_end':end,'history_first':str(dates[starts[0]-HISTORY].date()),
            'targets':[[str(dates[s].date()),str(dates[s+20].date())] for s in starts], **rounding}
        for fid in CASES+['H_historical']:
            path=SOURCE/fid/split/'paths.npz'
            summary=json.loads((path.parent/'summary.json').read_text())
            assert summary['status']=='complete' and sha(path)==summary['artifacts_sha256']['paths.npz']
            with np.load(path,allow_pickle=False) as z:
                assert z['tickers'].tolist()==TICKERS
                anchors=dates.get_indexer(pd.to_datetime(z['anchor_dates']))
                keep=(anchors-70>=lo)&(anchors+21<=hi)&(anchors>=0)
                indices=np.flatnonzero(keep)
                assert len(indices)>0
                history=np.stack([prices[a-70:a+1] for a in anchors[keep]])
                stored=np.stack([z[f'history_{k}'][z['history_index'][keep]] for k in ('close','high','low')],axis=-1)
                np.testing.assert_allclose(history[:,-63:],stored,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(history[:,-1,:,0],z['anchor_close'][keep])
                future=np.stack([z[k][keep] for k in ('close','high','low')],axis=-1)
                values=np.concatenate([history,future],axis=1)
                rounding=original.price_roundoff(values)
                name=f'{fid}_{split}'
                np.save(banks/f'{name}.npy',values)
                result['banks'][name]={'n':len(values),'original_retained':len(keep),
                    'excluded_insufficient_split_history':int((~keep).sum()),
                    'original_path_indices':indices.tolist(),'excluded_original_path_indices':np.flatnonzero(~keep).tolist(),
                    'candidate_ids':z['candidate_id'][keep].tolist(),'anchor_dates':z['anchor_dates'][keep].tolist(),
                    'source_paths':str(path),'source_paths_sha256':sha(path),
                    'sha256':sha(banks/f'{name}.npy'),**rounding}
    write_json(target,result)
    return result


class Controller(original.Controller):
    def __init__(self,real,synthetic,mode,case_id=0,pool_size=16,k_max=21,seed=0):
        self.k_max,self.seed=k_max,seed
        # Independent path/k RNGs. Paired real/synthetic slots share the k stream,
        # preserving exactly 50% source episodes/steps despite variable lengths.
        self.k_rngs=[np.random.default_rng(np.random.SeedSequence([seed,400,case_id,i%8])) for i in range(16)]
        super().__init__(real,synthetic,mode,case_id,pool_size)

    def draw_k(self,slot):
        return int(self.k_rngs[slot].integers(0,self.k_max+1))

    def replay_sample(self):
        members=self.members
        prices=np.stack([self.prices(source,index) for source,index in members])
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,500,self.case_id,self.generation]))
        ks=rng.integers(0,self.k_max+1,len(members))
        return prices,ks

    def state(self):
        return {**super().state(),'k_rngs':[r.bit_generator.state for r in self.k_rngs]}

    def restore(self,state):
        super().restore(state)
        for r,s in zip(self.k_rngs,state['k_rngs']):r.bit_generator.state=s


class PilotEnv(original.PilotEnv):
    def __init__(self,controller,slot=0):
        super().__init__(controller,slot)
        self.k=0;self.generation_at_start=controller.generation
        self.prefix_wealth=1.;self.k_counts=np.zeros(22,dtype=int)
        self.synthetic_days=0;self.real_days=0;self.prefix_days=0
        self.suffix_days=0;self.stale_pool_days=0;self.k_sum=0

    def _obs(self):
        obs=observations(self.controller.prices(self.source,self.index)[None],
                         self.day+K_MAX-self.k,self.weights[None])
        return {key:value[0] for key,value in obs.items()}

    def reset(self,seed=None,options=None):
        super(original.PilotEnv,self).reset(seed=seed)
        if seed is not None:
            self.controller.samplers[self.slot]=np.random.default_rng(np.random.SeedSequence([seed,200,self.controller.case_id,self.slot]))
            paired_seed=(int(seed)-self.slot)%(2**32)
            self.controller.k_rngs[self.slot]=np.random.default_rng(np.random.SeedSequence([paired_seed,400,self.controller.case_id,self.slot%8]))
        self.source,self.index=self.controller.pick(self.slot)
        self.k=self.controller.draw_k(self.slot)
        self.day,self.wealth,self.prefix_wealth=0,1.,1.
        self.weights=np.r_[1.,np.zeros(10)]
        self.generation_at_start=self.controller.generation
        return self._obs(),{}

    def rebind_idle(self):
        """Automatic reset selected an unused old-pool path; keep its drawn k."""
        if self.day==0:
            self.source,self.index=self.controller.pick(self.slot)
            self.generation_at_start=self.controller.generation

    def step_weights(self,target):
        if self.day>=self.k+21:raise RuntimeError('Episode already complete')
        prices=self.controller.prices(self.source,self.index)
        index=HISTORY-self.k+self.day
        before=self.wealth
        if self.day==self.k:self.prefix_wealth=self.wealth
        gross=np.r_[1.,prices[index,:,0]/prices[index-1,:,0]]
        factor,weights,cost,turnover=transact(self.weights,target,gross)
        change=float(target[1:].sum()-self.weights[1:].sum())
        self.wealth*=float(factor);self.weights=weights
        is_prefix=self.day<self.k
        self.prefix_days+=int(is_prefix);self.suffix_days+=int(not is_prefix)
        synthetic=bool(self.source==1 and not is_prefix)
        self.synthetic_days+=int(synthetic);self.real_days+=int(not synthetic)
        self.stale_pool_days+=int(self.generation_at_start<self.controller.generation)
        self.day+=1
        done=self.day==self.k+21
        reward=100*math.log(float(factor))
        info={'wealth':self.wealth,'previous_wealth':before,'cost_fraction':float(cost),
            'turnover':float(turnover),'exposure_change_from_trade':change,'source':self.source,
            'path_id':self.index,'k':self.k,'prefix':is_prefix,'pool_generation':self.generation_at_start}
        if done:
            self.completed[self.source]+=1;self.k_counts[self.k]+=1;self.k_sum+=self.k
            info['episode']={'r':100*math.log(self.wealth),'l':self.k+21,
                'suffix_r':100*math.log(self.wealth/self.prefix_wealth),'k':self.k}
        return self._obs(),reward,done,False,info

    def state(self):
        return {**super().state(),**{key:getattr(self,key) for key in [
            'k','generation_at_start','prefix_wealth','k_counts','synthetic_days','real_days',
            'prefix_days','suffix_days','stale_pool_days','k_sum']}}

    def restore(self,state):
        super().restore(state)
        for key in ['k','generation_at_start','prefix_wealth','synthetic_days','real_days',
                    'prefix_days','suffix_days','stale_pool_days','k_sum']:setattr(self,key,state[key])
        self.k_counts=np.array(state['k_counts'])


def evaluate(model,prices,ks,details=False):
    """Real trading through the prelude, with no reset at the synthetic boundary."""
    prices=np.asarray(prices);ks=np.asarray(ks,dtype=int)
    assert len(prices)==len(ks) and ((ks>=0)&(ks<=21)).all()
    wealth=np.ones(len(prices));prefix_wealth=np.ones(len(prices))
    weights=np.zeros((len(prices),11));weights[:,0]=1
    with torch.inference_mode():
        for day in range(int(ks.max())+21):
            active=np.flatnonzero(day<ks+21)
            boundary=active[ks[active]==day]
            prefix_wealth[boundary]=wealth[boundary]
            # Histories have different start offsets; observations stay causal.
            offsets=K_MAX-ks[active]+day
            windows=np.stack([prices[i,s:s+50] for i,s in zip(active,offsets)])
            obs={'state':(windows.transpose(0,3,2,1)/windows[:,-1,:,0][:,None,:,None]).astype(np.float32),
                 'last_action':weights[active].astype(np.float32)}
            action,_=model.predict(obs,deterministic=True)
            target=action_weights(action)
            indices=HISTORY-ks[active]+day
            gross=np.column_stack([np.ones(len(active)),prices[active,indices,:,0]/prices[active,indices-1,:,0]])
            factor,updated,_cost,_turnover=transact(weights[active],target,gross)
            weights[active]=updated;wealth[active]*=factor
    scores={'full':100*np.log(wealth),'prefix':100*np.log(prefix_wealth),
            'suffix':100*np.log(wealth/prefix_wealth)}
    return scores if details else scores['suffix']


def real_evaluation(model,prices):
    # Average over all k within each target episode; n remains the target count.
    scores=evaluate(model,np.repeat(prices,22,axis=0),np.tile(np.arange(22),len(prices)),True)
    return {key:value.reshape(len(prices),22).mean(axis=1) for key,value in scores.items()}


def fixed_ks(n):
    values=np.arange(n)%22
    np.random.default_rng(20261116).shuffle(values)
    return values


def validation_row(model,real,syn,train,syntrain,update):
    r=real_evaluation(model,real)
    s=evaluate(model,syn,fixed_ks(len(syn)),True) if syn is not None else {k:np.array([]) for k in ('suffix','full','prefix')}
    tr=evaluate(model,train,fixed_ks(len(train)),True)
    ts=evaluate(model,syntrain,fixed_ks(len(syntrain)),True) if syntrain is not None else {k:np.array([]) for k in ('suffix','full','prefix')}
    return {'update':update,'real':r['suffix'].tolist(),'synthetic':s['suffix'].tolist(),
        'real_full':r['full'].tolist(),'synthetic_full':s['full'].tolist(),
        'train_real':tr['suffix'].tolist(),'train_synthetic':ts['suffix'].tolist(),
        'train_real_full':tr['full'].tolist(),'train_synthetic_full':ts['full'].tolist()}


def worker(job_path,stop_after=None):
    job_path=Path(job_path);out=job_path.parent;config=json.loads(job_path.read_text())
    torch.set_num_threads(config.get('threads',2));torch.set_num_interop_threads(1)
    banks=Path(config['banks'])
    real=np.load(banks/'real_train.npy',mmap_mode='r')
    val=np.load(banks/'real_validation.npy',mmap_mode='r')
    train=real[np.linspace(0,len(real)-1,64,dtype=int)]
    syn=synval=syntrain=None
    if config['group']!='R':
        fid='H_historical' if config['group']=='H' else config['case']
        syn=np.load(banks/f'{fid}_train.npy',mmap_mode='r')
        synval=np.load(banks/f'{fid}_validation.npy',mmap_mode='r')
        syntrain=syn[np.sort(np.random.default_rng(20261116).choice(len(syn),64,replace=False))]
    controller=Controller(real,syn,config['group'],config['case_id'],config.get('pool_size',16),config['k_max'])
    envs=[PilotEnv(controller,i) for i in range(16)]
    vec=DummyVecEnv([lambda e=e:e for e in envs])
    config_hash=digest(config);statepath=out/'status.json'
    if statepath.exists():
        prior=json.loads(statepath.read_text());assert prior['config_sha256']==config_hash
        if prior['status']=='complete':return
    write_json(statepath,{'status':'running','config':config,'config_sha256':config_hash,'updated_utc':now()})
    resume=out/'resume.json';records=[];evaluations=[]
    if resume.exists():
        state=json.loads(resume.read_text())
        controller.restore(state['controller'])
        for env,value in zip(envs,state['envs']):env.restore(value)
        model=PPO.load(out/state['checkpoint'],env=vec,device='cpu',force_reset=False)
        model._last_obs={key:np.stack([env._obs()[key] for env in envs]) for key in ('state','last_action')}
        model._last_episode_starts=np.array(state['last_episode_starts'],dtype=bool)
        vec._seeds=[None]*16
        torch.set_rng_state(torch.tensor(state['torch_rng'],dtype=torch.uint8))
        ns=state['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        ps=state['python_rng'];random.setstate((ps[0],tuple(ps[1]),ps[2]))
        records=[r for r in json.loads((out/'updates.json').read_text()) if r['update']<=state['update']]
        evaluations=[r for r in json.loads((out/'validation_returns.json').read_text()) if r['update']<=state['update']]
        initial=json.loads((out/'initialization.json').read_text())
        best=state['best'];age=state['pool_age'];plateau=state['plateau'];start=state['update']+1
        before=np.array(evaluations[-1]['real'])
    else:
        model=PPO('MultiInputPolicy',vec,seed=0,device='cpu',n_steps=N_STEPS,batch_size=504,n_epochs=10,
            learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,verbose=0,
            policy_kwargs={'features_extractor_class':EIIEWindowExtractor,'net_arch':dict(pi=[64],vf=[64])})
        initial={'seed':0,'policy_sha256':parameter_hash(model),'parameter_count':sum(p.numel() for p in model.policy.parameters())}
        write_json(out/'initialization.json',initial)
        first=validation_row(model,val,synval,train,syntrain,0);evaluations=[first]
        before=np.array(first['real']);best=float(before.mean());age=0;plateau=0;start=1
        model.save(out/'initial_model');model.save(out/'best_model')
    model.set_logger(configure(str(out/'ppo'),['csv']))
    started=time.monotonic()
    final=min(config['updates'],stop_after) if stop_after is not None else config['updates']
    for update in range(start,final+1):
        replay=controller.replay_sample() if config['group']=='D' else None
        replay_before=evaluate(model,*replay) if replay is not None else None
        tick=time.monotonic()
        model.learn(total_timesteps=BATCH,reset_num_timesteps=False,progress_bar=False,log_interval=1)
        training_seconds=time.monotonic()-tick
        assert model.num_timesteps==update*BATCH
        tick=time.monotonic()
        row=validation_row(model,val,synval,train,syntrain,update)
        after=np.array(row['real']);change=after-before
        replay_delta=evaluate(model,*replay)-replay_before if replay is not None else np.array([])
        validation_seconds=time.monotonic()-tick
        age+=1;rotate=False;reason='continue'
        if config['group']=='D':
            if config.get('adaptive'):
                thresholds=config['adaptive'];plateau=plateau+1 if change.mean()<=thresholds['improvement_threshold'] else 0
                if age>=thresholds['minimum_repeats'] and change.mean() < -thresholds['regression_threshold']:
                    reason='validation_regression';rotate=True
                elif age>=thresholds['minimum_repeats'] and plateau>=2:
                    reason='plateau';rotate=True
                elif age>=thresholds['maximum_repeats']:
                    reason='maximum_repeats';rotate=True
            elif age>=config['repeat']:
                reason='fixed_repeat_limit';rotate=True
        episodes=list(model.ep_info_buffer)
        counts=np.sum([env.k_counts for env in envs],axis=0)
        record={'update':update,'steps':model.num_timesteps,'real_validation_score':float(after.mean()),
            'synthetic_validation_score':float(np.mean(row['synthetic'])) if row['synthetic'] else None,
            'validation_gain':float(change.mean()),'validation_gain_se':se(change),
            'replay_gain':float(replay_delta.mean()) if len(replay_delta) else None,
            'replay_gain_se':se(replay_delta) if len(replay_delta) else None,
            'pool_generation':controller.generation,'pool_age':age,'switch_reason':reason,
            'unfinished_episodes_at_requested_switch':sum(env.day>0 for env in envs) if rotate else 0,
            'training_seconds':training_seconds,'validation_seconds':validation_seconds,
            'completed_real_episodes':int(sum(env.completed[0] for env in envs)),
            'completed_synthetic_episodes':int(sum(env.completed[1] for env in envs)),
            'actual_real_price_steps':sum(env.real_days for env in envs),
            'actual_synthetic_price_steps':sum(env.synthetic_days for env in envs),
            'prefix_steps':sum(env.prefix_days for env in envs),'suffix_steps':sum(env.suffix_days for env in envs),
            'stale_pool_continuation_steps':sum(env.stale_pool_days for env in envs),
            'completed_k_counts':counts.tolist(),'mean_completed_k':float(np.dot(counts,np.arange(22))/counts.sum()),
            'rollout_reward_mean':float(np.mean([e['r'] for e in episodes])),
            'rollout_suffix_reward_mean':float(np.mean([e['suffix_r'] for e in episodes])),
            'approx_kl':float(model.logger.name_to_value.get('train/approx_kl',0))}
        assert record['actual_real_price_steps']+record['actual_synthetic_price_steps']==model.num_timesteps
        if config['group']!='R':
            assert record['completed_real_episodes']==record['completed_synthetic_episodes']
            assert all(envs[i].day==envs[i+8].day and envs[i].k==envs[i+8].k for i in range(8))
        records.append(record);evaluations.append(row)
        if after.mean()>best:
            best=float(after.mean());model.save(out/'best_model')
        if rotate:
            controller.rotate();age=0;plateau=0
            # Do not truncate, reset wealth, or discard partial episodes.
            for env in envs:env.rebind_idle()
            model._last_obs={key:np.stack([env._obs()[key] for env in envs]) for key in ('state','last_action')}
        tick=time.monotonic();version=out/f'checkpoint_{update:03d}.zip'
        model.save(version);shutil.copy2(version,out/'latest_model.zip')
        record['checkpoint_seconds']=time.monotonic()-tick
        ns=np.random.get_state()
        write_json(out/'updates.json',records);pd.DataFrame(records).to_csv(out/'updates.csv',index=False)
        write_json(out/'validation_returns.json',evaluations)
        write_json(resume,{'update':update,'checkpoint':version.name,'controller':controller.state(),
            'envs':[env.state() for env in envs],'last_episode_starts':model._last_episode_starts,
            'torch_rng':torch.get_rng_state().tolist(),'numpy_rng':ns,'python_rng':random.getstate(),
            'best':best,'pool_age':age,'plateau':plateau})
        before=after
        write_json(statepath,{'status':'running','config':config,'config_sha256':config_hash,'update':update,
            'steps':model.num_timesteps,'best_validation_score':best,'initialization':initial,'updated_utc':now()})
        print(f'{config["name"]} update={update}/{config["updates"]} val={after.mean():.5f} k={record["mean_completed_k"]:.3f} switch={reason}',flush=True)
    complete=model.num_timesteps==config['updates']*BATCH
    write_json(statepath,{'status':'complete' if complete else 'checkpoint_ready','config':config,
        'config_sha256':config_hash,'steps':model.num_timesteps,'best_validation_score':best,
        'last_validation_score':float(before.mean()),'initialization':initial,
        'wall_seconds_this_process':time.monotonic()-started,'completed_utc':now()})
    vec.close()


def make_job(out,name,case,group,updates,repeat=1,pool=16,adaptive=None):
    folder=Path(out)/'jobs'/name;folder.mkdir(parents=True,exist_ok=True)
    event=case.rsplit('_',2)[0] if case in CASES else case
    config={'name':name,'case':case,'case_id':EVENTS.index(event),'group':group,'updates':updates,
        'repeat':repeat,'pool_size':pool,'banks':str(Path(out)/'banks'),'threads':2,'seed':0,
        'k_max':21,'commission':FEE,'source_code_sha256':sha(__file__)}
    if adaptive is not None:config['adaptive']=adaptive
    path=folder/'job.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:write_json(path,config)
    return path


def run_jobs(out,jobs,workers,stage):
    write_json(Path(out)/'status.json',{'status':'running','stage':stage,'jobs':[str(p.parent) for p in jobs],'updated_utc':now()})
    def one(path):
        status=path.parent/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='complete':return str(path.parent),'cached'
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
        with (path.parent/'training.log').open('a') as log:
            command=[sys.executable,'-u','-m','finrl.experiments.timediff_random_k','worker','--job',str(path)]
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{path.parent}: see training.log')
        return str(path.parent),'complete'
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(one,path) for path in jobs]
        for future in as_completed(futures):print(stage,*future.result(),flush=True)


def read_scores(path,updates=8):
    rows=json.loads((path.parent/'validation_returns.json').read_text())
    return np.mean([r['real'] for r in rows if max(1,updates-3)<=r['update']<=updates],axis=0)


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
              'threshold_units':'21-day suffix reward after trading a random real prelude; no boundary reset',
              'threshold_estimation':'median / 90th percentile of paired real-2018 validation update standard errors',
              'best_fixed_repeat':best_repeat,'best_raw_pool':best_pool,'larger_pool_triggered':overfit,
              'calibration_interactions_per_candidate':8*BATCH,
              'selection_score':'mean paired real-2018 validation gain over updates 5-8 versus C at the same step',
              'pooled_case_score':float(pool_vectors[chosen_pool].mean()),'formal_parameters_locked':False,
              'k_max':21,'pool_switch_policy':'finish in-flight episodes; new pool applies at the next natural reset (at most 41 extra steps per env)', 'limitation':'seed=0; 2018 validation only; no 2019/event performance used for calibration'}
    pd.DataFrame([{k:v for k,v in r.items() if k!='episode_differences'} for r in comparisons]).to_csv(out/'D_calibration_candidates.csv',index=False)
    write_json(out/'D_calibration_evidence.json',comparisons)
    write_json(out/'D_selected_parameters.json',settings)
    adaptive={k:settings[k] for k in ('minimum_repeats','maximum_repeats','improvement_threshold','regression_threshold')}
    final=[make_job(out,f'{case}_D_adaptive',case,'D',32,pool=chosen_pool,adaptive=adaptive) for case in CASES]
    run_jobs(out,final,workers,'step3_adaptive_D_validation')
    return settings


def run(out,workers=8):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cpus=list(range(9))+list(range(12,21))
    os.sched_setaffinity(0,cpus)
    os.nice(5)
    plan={'created_utc':now(),'scope':'random k=0..21; seed=0 development rerun only',
        'source':str(SOURCE),'initial_updates':32,'calibration_updates':8,'n_envs':16,
        'n_steps':504,'total_interactions_per_primary_model':32*BATCH,
        'workers':workers,'torch_threads_per_worker':2,'cpu_affinity':cpus,
        'cpu_cap_fraction':len(cpus)/24,'device':'cpu',
        'gpu_reason':'installed torch CUDA13 requires a newer driver than installed driver570; CPU parallelism used',
        'k_policy':'iid discrete uniform 0..21 on each reset; real/synthetic paired slots share k stream',
        'partial_episodes':'continue across rollouts; D new pool starts at next natural episode reset',
        'selection_metric':'mean of final 21 traded-day log rewards after real prelude; average all k within each real validation target',
        'data_selection':'all 71 history +21 future dates inside each split; excluded IDs retained in manifest',
        'test':'2019 diagnostic evaluated only after D selection; no event test or formal seeds',
        'code_sha256':sha(__file__)}
    if (out/'plan.json').exists():
        prior=json.loads((out/'plan.json').read_text());assert prior['code_sha256']==plan['code_sha256']
    else:write_json(out/'plan.json',plan)
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    try:
        build_banks(out)
        jobs=[]
        for event in EVENTS:
            for group in ('R','H'):jobs.append(make_job(out,f'{event}_{group}',event,group,32))
        for case in CASES:jobs.append(make_job(out,f'{case}_C',case,'C',32))
        run_jobs(out,jobs,workers,'primary_R_H_C')
        settings=calibration(out,workers)
        write_json(out/'training_complete.json',{'status':'complete','completed_utc':now(),'settings':settings})
        write_json(out/'status.json',{'status':'running','stage':'diagnostic_curves_and_audit','updated_utc':now()})
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/finalize_timediff_random_k.py'),
                        '--out',str(out),'--workers',str(workers)],cwd=ROOT,check=True)
    except Exception as error:
        write_json(out/'status.json',{'status':'failed','error':str(error),'updated_utc':now()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p=sub.add_parser('worker');p.add_argument('--job',type=Path,required=True);p.add_argument('--stop-after-update',type=int)
    p=sub.add_parser('prepare');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='worker':worker(args.job,args.stop_after_update)
    elif args.command=='prepare':build_banks(args.out)
    else:run(args.out,args.workers)
