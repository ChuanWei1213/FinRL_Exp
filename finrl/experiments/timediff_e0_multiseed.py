"""Three E0 conditions, frozen TimeDiff lambda=.5 data, PPO seeds1..10.

Train pool remains2005-2017. Validation targets2018-2019; past-only history
can precede2018. Seed0 calibrates D before all main runs. No event selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from finrl.experiments import timediff_random_k as k

ROOT,SOURCE=k.ROOT,k.SOURCE
PREVIOUS=Path('/mnt/sda/cwei1213/timediff_finrl/random_k_20261116')
DEFAULT_OUT=Path('/mnt/sda/cwei1213/timediff_finrl/e0_10seeds_val2018_2019_20261116')
EVENTS=k.EVENTS
CASES=[f'{event}_E0_f0' for event in EVENTS]
SEEDS=list(range(1,11))
N_ENVS,N_STEPS,BATCH,FEE=16,504,8064,.0025
now,sha,write_json,digest,se=k.now,k.sha,k.write_json,k.digest,k.se
EIIEWindowExtractor,parameter_hash=k.EIIEWindowExtractor,k.parameter_hash
evaluate,validation_row=k.evaluate,k.validation_row
PilotEnv=k.PilotEnv


class Controller(k.Controller):
    """Seed every independent stream, including D pool selection."""
    def __init__(self,real,synthetic,mode,case_id=0,pool_size=16,k_max=21,seed=0):
        super().__init__(real,synthetic,mode,case_id,pool_size,k_max,seed)
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,100,case_id]))
        self.samplers=[np.random.default_rng(np.random.SeedSequence([seed,200,case_id,i])) for i in range(16)]
        self.generation=0;self.members=[];self.rotate()


def build_banks(out):
    out=Path(out);banks=out/'banks';banks.mkdir(parents=True,exist_ok=True)
    path=banks/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((PREVIOUS/'banks/manifest.json').read_text())
    source=json.loads((SOURCE/'manifest.json').read_text())
    assert source['generator_metadata']['args']['lambda_condition']==.5
    assert sha(source['data'])==source['data_sha256']
    assert sha(source['checkpoint'])==source['checkpoint_sha256']
    result={'created_utc':now(),'generator_lambda':.5,'generator_checkpoint_sha256':source['checkpoint_sha256'],
        'price_source':source['data'],'price_source_sha256':source['data_sha256'],
        'cases':CASES,'train':'exact same bank arrays as preceding random-k experiment,2005-2017',
        'validation_targets':['2018-01-01','2019-12-31'],
        'validation_history':'past-only history may precede2018; none of the scored future targets enter train',
        'synthetic_validation':'2018 retained validation +2019 formerly named test; both now validation',
        'test':'actual first21 trading days of the three E0 events, not built/read by training workers',
        'banks':{}}
    for name in ['real_train']+[f'{fid}_train' for fid in CASES+['H_historical']]:
        src=PREVIOUS/'banks'/f'{name}.npy';dest=banks/f'{name}.npy'
        assert sha(src)==old['banks'][name]['sha256']
        shutil.copyfile(src,dest)
        result['banks'][name]={**old['banks'][name],'source_bank':str(src),'identical_to_previous_training':True}
    frame=pd.read_csv(source['data'],parse_dates=['date'])
    frame=frame[frame.date<='2019-12-31']
    wide={key:frame.pivot(index='date',columns='tic',values=key)[k.TICKERS] for key in ('close','high','low')}
    dates=wide['close'].index
    prices=np.stack([wide[key].to_numpy() for key in ('close','high','low')],axis=-1)
    eligible=np.flatnonzero((dates>=pd.Timestamp('2018-01-01'))&(dates<=pd.Timestamp('2019-12-31')))
    lo,hi=int(eligible[0]),int(eligible[-1])
    starts=np.arange(lo,hi-20+1,21)
    # Preserve complete21-day windows and cover the requested validation tail.
    # The final appended window overlaps the preceding window by one day.
    if starts[-1]+20<hi:starts=np.r_[starts,hi-20]
    values=np.stack([prices[s-71:s+21] for s in starts])
    rounding=k.original.price_roundoff(values)
    np.save(banks/'real_validation.npy',values)
    result['banks']['real_validation']={'n':len(values),'sha256':sha(banks/'real_validation.npy'),
        'targets':[[str(dates[s].date()),str(dates[s+20].date())] for s in starts],
        'first_history':str(dates[starts[0]-71].date()),'target_days_scored':len(starts)*21,
        'unique_target_days':int(hi-lo+1),'repeated_target_days':int(len(starts)*21-(hi-lo+1)),
        'tail_days_insufficient_for_21':int(hi-(starts[-1]+20)),**rounding}
    for fid in CASES+['H_historical']:
        chunks=[];sources=[]
        for original_split in ('validation','test'):
            src=SOURCE/fid/original_split/'paths.npz'
            summary=json.loads((src.parent/'summary.json').read_text())
            assert summary['status']=='complete' and sha(src)==summary['artifacts_sha256']['paths.npz']
            with np.load(src,allow_pickle=False) as z:
                assert z['tickers'].tolist()==k.TICKERS
                anchors=dates.get_indexer(pd.to_datetime(z['anchor_dates']))
                assert (anchors>=70).all() and ((anchors+1)>=lo).all() and ((anchors+21)<=hi).all()
                history=np.stack([prices[a-70:a+1] for a in anchors])
                recorded=np.stack([z[f'history_{key}'][z['history_index']] for key in ('close','high','low')],axis=-1)
                np.testing.assert_allclose(history[:,-63:],recorded,rtol=1e-12,atol=1e-12)
                future=np.stack([z[key] for key in ('close','high','low')],axis=-1)
                chunk=np.concatenate([history,future],axis=1)
                k.original.price_roundoff(chunk);chunks.append(chunk)
                sources.append({'original_split':original_split,'new_split':'validation','path':str(src),
                    'sha256':sha(src),'n':len(chunk),'candidate_ids':z['candidate_id'].tolist(),
                    'anchor_dates':z['anchor_dates'].tolist(),'background_ids':z['background_id'].tolist()})
        values=np.concatenate(chunks);assert len(values)==400
        name=f'{fid}_validation';np.save(banks/f'{name}.npy',values)
        result['banks'][name]={'n':len(values),'sha256':sha(banks/f'{name}.npy'),'sources':sources}
    write_json(path,result)
    return result


class BatchVecEnv(DummyVecEnv):
    """Batch the same16 accounts; retain SB3 auto-reset/terminal-observation API."""
    def batched_obs(self):
        windows=np.stack([e.controller.prices(e.source,e.index)[21-e.k+e.day:71-e.k+e.day] for e in self.envs])
        return {'state':(windows.transpose(0,3,2,1)/windows[:,-1,:,0][:,None,:,None]).astype(np.float32),
                'last_action':np.stack([e.weights for e in self.envs]).astype(np.float32)}

    def step_wait(self):
        previous=np.stack([e.weights for e in self.envs])
        targets=k.action_weights(self.actions)
        ratios=np.stack([e.controller.prices(e.source,e.index)[71-e.k+e.day,:,0]/
                         e.controller.prices(e.source,e.index)[70-e.k+e.day,:,0] for e in self.envs])
        gross=np.column_stack([np.ones(len(self.envs)),ratios])
        factors,weights,costs,turnover=k.transact(previous,targets,gross)
        for i,e in enumerate(self.envs):
            before=e.wealth
            if e.day==e.k:e.prefix_wealth=e.wealth
            e.wealth*=float(factors[i]);e.weights=weights[i]
            prefix=e.day<e.k;synthetic=bool(e.source==1 and not prefix)
            e.prefix_days+=int(prefix);e.suffix_days+=int(not prefix)
            e.synthetic_days+=int(synthetic);e.real_days+=int(not synthetic)
            e.stale_pool_days+=int(e.generation_at_start<e.controller.generation)
            e.day+=1;done=e.day==e.k+21
            self.buf_rews[i]=100*np.log(float(factors[i]));self.buf_dones[i]=done
            info={'wealth':e.wealth,'previous_wealth':before,'cost_fraction':float(costs[i]),
                'turnover':float(turnover[i]),'exposure_change_from_trade':float(targets[i,1:].sum()-previous[i,1:].sum()),
                'source':e.source,'path_id':e.index,'k':e.k,'prefix':prefix,
                'pool_generation':e.generation_at_start,'TimeLimit.truncated':False}
            if done:
                e.completed[e.source]+=1;e.k_counts[e.k]+=1;e.k_sum+=e.k
                info['episode']={'r':100*np.log(e.wealth),'l':e.k+21,
                                 'suffix_r':100*np.log(e.wealth/e.prefix_wealth),'k':e.k}
            self.buf_infos[i]=info
        obs=self.batched_obs()
        for i,e in enumerate(self.envs):
            if self.buf_dones[i]:
                self.buf_infos[i]['terminal_observation']={key:value[i].copy() for key,value in obs.items()}
                reset_obs,self.reset_infos[i]=e.reset()
                for key in obs:obs[key][i]=reset_obs[key]
        for key in obs:self.buf_obs[key][:]=obs[key]
        return self._obs_from_buf(),np.copy(self.buf_rews),np.copy(self.buf_dones),deepcopy(self.buf_infos)


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
    controller=Controller(real,syn,config['group'],config['case_id'],config.get('pool_size',16),config['k_max'],seed=config['seed'])
    envs=[PilotEnv(controller,i) for i in range(16)]
    vec=BatchVecEnv([lambda e=e:e for e in envs])
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
        model=PPO('MultiInputPolicy',vec,seed=config['seed'],device='cpu',n_steps=N_STEPS,batch_size=504,n_epochs=10,
            learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,verbose=0,
            policy_kwargs={'features_extractor_class':EIIEWindowExtractor,'net_arch':dict(pi=[64],vf=[64])})
        initial={'seed':config['seed'],'policy_sha256':parameter_hash(model),'parameter_count':sum(p.numel() for p in model.policy.parameters())}
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


def make_job(out,name,case,group,updates,repeat=1,pool=16,adaptive=None,seed=0,phase='pilot',threads=2):
    folder=Path(out)/'jobs'/f'{name}_seed{seed:02d}';folder.mkdir(parents=True,exist_ok=True)
    event=case.rsplit('_',2)[0] if case in CASES else case
    config={'name':folder.name,'case':case,'case_id':EVENTS.index(event),'group':group,'updates':updates,
        'repeat':repeat,'pool_size':pool,'banks':str(Path(out)/'banks'),'threads':threads,'seed':seed,
        'phase':phase,'k_max':21,'commission':FEE,'backend':'BatchVecEnv','source_code_sha256':sha(__file__),
        'bank_manifest_sha256':sha(Path(out)/'banks/manifest.json')}
    if adaptive is not None:config['adaptive']=adaptive
    path=folder/'job.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:write_json(path,config)
    return path


def run_jobs(out,jobs,workers,stage):
    write_json(Path(out)/'status.json',{'status':'running','stage':stage,'jobs':[str(p.parent) for p in jobs],
                                      'updated_utc':now()})
    def one(path):
        status=path.parent/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='complete':return path.parent.name,'cached'
        config=json.loads(path.read_text());threads=str(config['threads'])
        env={**os.environ,'OMP_NUM_THREADS':threads,'MKL_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
        cmd=[sys.executable,'-u','-m','finrl.experiments.timediff_e0_multiseed','worker','--job',str(path)]
        with (path.parent/'training.log').open('a') as handle:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        return path.parent.name,'complete'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,path) for path in jobs]
        for future in as_completed(futures):print(stage,*future.result(),flush=True)


def read_scores(path):
    rows=json.loads((path.parent/'validation_returns.json').read_text())
    return np.mean([row['real'] for row in rows if 5<=row['update']<=8],axis=0)


def calibrate(out,workers,threads):
    lock=out/'D_parameters_locked.json'
    if lock.exists():return json.loads(lock.read_text())
    cpaths={case:make_job(out,f'{case}_C',case,'C',8,threads=threads) for case in CASES}
    run_jobs(out,list(cpaths.values()),workers,'pilot_seed0_C')
    grid={}
    for case in CASES:
        for repeat in (1,2,4,8):
            grid[case,repeat,16]=make_job(out,f'{case}_D_repeat{repeat}_pool16',case,'D',8,repeat,threads=threads)
    run_jobs(out,list(grid.values()),workers,'pilot_seed0_D_repeats')
    comparisons=[];errors=[];overfit=False
    for (case,repeat,pool),path in grid.items():
        delta=read_scores(path)-read_scores(cpaths[case])
        updates=json.loads((path.parent/'updates.json').read_text())
        for row in updates:
            errors.append(row['validation_gain_se'])
            if row['replay_gain']>row['replay_gain_se'] and row['validation_gain'] < -row['validation_gain_se']:overfit=True
        comparisons.append({'case':case,'repeat':repeat,'pool_size':pool,'paired_gain_vs_C':float(delta.mean()),
                            'paired_se':se(delta),'target_differences':delta.tolist()})
    by_repeat={repeat:np.mean([row['target_differences'] for row in comparisons if row['repeat']==repeat],axis=0) for repeat in (1,2,4,8)}
    best_repeat=max(by_repeat,key=lambda repeat:float(by_repeat[repeat].mean()))
    if overfit:
        larger={}
        for case in CASES:
            for pool in (32,64):
                larger[case,best_repeat,pool]=make_job(out,f'{case}_D_repeat{best_repeat}_pool{pool}',case,'D',8,best_repeat,pool,threads=threads)
        run_jobs(out,list(larger.values()),workers,'pilot_seed0_D_larger_pools')
        for (case,repeat,pool),path in larger.items():
            delta=read_scores(path)-read_scores(cpaths[case])
            comparisons.append({'case':case,'repeat':repeat,'pool_size':pool,'paired_gain_vs_C':float(delta.mean()),
                                'paired_se':se(delta),'target_differences':delta.tolist()})
    pool_vectors={}
    for pool in (16,32,64):
        values=[row['target_differences'] for row in comparisons if row['repeat']==best_repeat and row['pool_size']==pool]
        if values:pool_vectors[pool]=np.mean(values,axis=0)
    best_pool=max(pool_vectors,key=lambda pool:float(pool_vectors[pool].mean()))
    chosen_pool=min(pool for pool,value in pool_vectors.items() if
        np.mean(pool_vectors[best_pool]-value)<=se(pool_vectors[best_pool]-value)+1e-12)
    minimum=min(repeat for repeat,value in by_repeat.items() if repeat<=best_repeat and
        np.mean(by_repeat[best_repeat]-value)<=se(by_repeat[best_repeat]-value)+1e-12)
    settings={'created_utc':now(),'selection_seed':0,'main_seeds':SEEDS,'validation_target_bounds':['2018-01-01','2019-12-31'],
        'environment_pool_size':chosen_pool,'minimum_repeats':minimum,'maximum_repeats':best_repeat,
        'improvement_threshold':max(float(np.median(errors)),1e-10),
        'regression_threshold':max(float(np.quantile(errors,.9)),1e-10),'plateau_patience':2,
        'best_fixed_repeat':best_repeat,'best_raw_pool':best_pool,'larger_pool_triggered':overfit,
        'pooled_gain_vs_C':float(pool_vectors[chosen_pool].mean()),
        'selection_metric':'mean paired21-day suffix reward over updates5..8; average cases before validation-target SE',
        'locked_before_main_seeds':True,'event_test_used_for_calibration':False,
        'validation_SE_is_selection_heuristic':'24 target windows; final two overlap one day; no market-independence claim',
        'pool_switch':'finish in-flight episodes; next natural reset, at most41 further steps per env'}
    pd.DataFrame([{key:value for key,value in row.items() if key!='target_differences'} for row in comparisons]).to_csv(out/'D_calibration_candidates.csv',index=False)
    write_json(out/'D_calibration_evidence.json',comparisons)
    write_json(lock,settings)
    return settings


def run(out,workers=8,threads=2):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cpus=list(range(9))+list(range(12,21));os.sched_setaffinity(0,cpus);os.nice(5)
    plan={'created_utc':now(),'cases':CASES,'main_seeds':SEEDS,'calibration_seed':0,
        'train_target_years':[2005,2017],'validation_target_dates':['2018-01-01','2019-12-31'],
        'test':'corresponding actual first21 trading days; cash start; onlyE0; no full-event static tests',
        'groups':['R','H','C','D'],'main_jobs':120,'main_updates':32,'interactions_per_main_job':32*BATCH,
        'pilot_updates':8,'k_max':21,'n_envs':16,'n_steps':504,'ppo_batch_size':504,'ppo_n_epochs':10,
        'workers':workers,'threads_per_worker':threads,'cpu_affinity':cpus,'cpu_cap_fraction':.75,
        'device':'cpu','backend':'BatchVecEnv','backend_validation':'matched reference accounts through reset/pool changes',
        'fee':FEE,'generator_lambda':.5,'synthetic_generation_seeds':'fixed existing paths shared across PPO seeds',
        'best_checkpoint_rule':'earliest maximum real validation suffix reward, including initialization',
        'secondary_checkpoint_rule':'final update32 sensitivity fixed before test',
        'uncertainty':'10 PPO seeds on the same three event paths; not10 independent markets or generator samples',
        'prior_seed0_event_results_previously_inspected':True,
        'module_sha256':sha(__file__),'random_k_dependency_sha256':sha(k.__file__),
        'original_helpers_sha256':sha(k.original.__file__)}
    if (out/'plan.json').exists():
        previous=json.loads((out/'plan.json').read_text());assert previous['module_sha256']==plan['module_sha256']
    else:write_json(out/'plan.json',plan)
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    try:
        build_banks(out)
        settings=calibrate(out,workers,threads)
        adaptive={key:settings[key] for key in ['minimum_repeats','maximum_repeats','improvement_threshold','regression_threshold']}
        jobs=[]
        for seed in SEEDS:
            for event,case in zip(EVENTS,CASES):
                for group in ['R','H','C','D']:
                    logical=f'{event}_{group}' if group in ('R','H') else f'{case}_{group}'
                    jobs.append(make_job(out,logical,event if group in ('R','H') else case,group,32,
                        pool=settings['environment_pool_size'] if group=='D' else 16,
                        adaptive=adaptive if group=='D' else None,seed=seed,phase='main',threads=threads))
        assert len(jobs)==120
        write_json(out/'main_jobs.json',[str(path) for path in jobs])
        run_jobs(out,jobs,workers,'main_seeds1_to10')
        write_json(out/'training_complete.json',{'status':'complete','completed_utc':now(),'main_jobs':120})
        write_json(out/'status.json',{'status':'running','stage':'locked_checkpoint_event_test','updated_utc':now()})
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/finalize_timediff_e0_multiseed.py'),
                        '--out',str(out),'--workers',str(workers)],cwd=ROOT,check=True)
    except Exception as error:
        write_json(out/'status.json',{'status':'failed','error':str(error),'updated_utc':now()})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('run');a.add_argument('--out',type=Path,default=DEFAULT_OUT);a.add_argument('--workers',type=int,default=8);a.add_argument('--threads',type=int,default=2)
    a=sub.add_parser('worker');a.add_argument('--job',type=Path,required=True);a.add_argument('--stop-after-update',type=int)
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    if args.command=='run':run(args.out,args.workers,args.threads)
    elif args.command=='worker':worker(args.job,args.stop_after_update)
    else:build_banks(args.out)
