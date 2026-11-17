"""Recover source labels for saved stochastic training episode buffers.

The episode dictionaries omit source, but the source slot and independent k RNG
determine the exact chronological completion order. Verify every saved buffer's
k/length, cumulative source/k counts, final per-slot state and RNG, and original
mixed reward. No policy inference, new rollouts, or parameter updates are used.
"""
from pathlib import Path
from functools import lru_cache
import base64,hashlib,json,pickle,zipfile
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
STUDY=P.parent

@lru_cache(maxsize=None)
def chronology(seed,case_id):
    events=[];final=[]
    for slot in range(16):
        rng=np.random.default_rng(np.random.SeedSequence([seed,400,case_id,slot%8]))
        t=0;k=int(rng.integers(0,22))
        while t+k+21<=32*504:
            t+=k+21;events.append((t,slot,k));k=int(rng.integers(0,22))
        final.append({'day':32*504-t,'k':k,'rng':rng.bit_generator.state})
    return np.array(sorted(events),dtype=np.int64),final

def decode(value):
    return pickle.loads(base64.b64decode(value[':serialized:']))

jobs=json.loads((STUDY/'main_jobs.json').read_text());assert len(jobs)==120
source_inputs={};points=[];max_mixed_error=0.;matched_episode_entries=0
for number,job_path in enumerate(jobs,1):
    job_path=Path(job_path);job=json.loads(job_path.read_text());folder=job_path.parent
    assert job['phase']=='main' and job['seed'] in range(1,11) and job['updates']==32 and job['k_max']==21
    event=job['case'].replace('_E0_f0','');group=job['group']
    schedule,final=chronology(job['seed'],job['case_id'])
    times,slots,ks=schedule.T
    sources=np.zeros(len(schedule),dtype=int) if group=='R' else (slots>=8).astype(int)
    updates=pd.read_csv(folder/'updates.csv').sort_values('update')
    assert updates['update'].tolist()==list(range(1,33))
    for u in updates.itertuples():
        step=int(u.update)*504;stop=int(np.searchsorted(times,step,side='right'));start=max(0,stop-100)
        expected=schedule[start:stop];labels=sources[start:stop]
        path=folder/f'checkpoint_{u.update:03d}.zip'
        with zipfile.ZipFile(path) as z:payload=z.read('data');data=json.loads(payload)
        source_inputs[str(path)+'::data']=hashlib.sha256(payload).hexdigest()
        assert data['num_timesteps']==u.steps==u.update*8064 and data['n_steps']==504 and data['n_envs']==16
        episodes=list(decode(data['ep_info_buffer']))
        assert len(episodes)==len(expected)==100
        np.testing.assert_array_equal([e['k'] for e in episodes],expected[:,2])
        np.testing.assert_array_equal([e['l'] for e in episodes],expected[:,2]+21)
        np.testing.assert_array_equal(decode(data['_last_episode_starts']),np.isin(np.arange(16),slots[times==step]))
        np.testing.assert_array_equal(np.bincount(sources[:stop],minlength=2),[u.completed_real_episodes,u.completed_synthetic_episodes])
        np.testing.assert_array_equal(np.bincount(ks[:stop],minlength=22),json.loads(u.completed_k_counts))
        rewards=np.array([e['r'] for e in episodes],dtype=float)
        suffix=np.array([e['suffix_r'] for e in episodes],dtype=float)
        np.testing.assert_allclose(rewards.mean(),u.rollout_reward_mean,atol=2e-12,rtol=0)
        np.testing.assert_allclose(suffix.mean(),u.rollout_suffix_reward_mean,atol=2e-12,rtol=0)
        max_mixed_error=max(max_mixed_error,abs(float(rewards.mean()-u.rollout_reward_mean)))
        weighted=0.
        for source,name in [(0,'real'),(1,'synthetic')]:
            take=labels==source;n=int(take.sum())
            mean=float(rewards[take].mean()) if n else np.nan
            tail=float(suffix[take].mean()) if n else np.nan
            points.append({'seed':job['seed'],'event':event,'group':group,'update':u.update,
                'steps':u.steps,'source':name,'n_episodes':n,'buffer_size':len(episodes),
                'episode_reward_mean':mean,'suffix_reward_mean':tail,
                'prefix_reward_mean':mean-tail if n else np.nan})
            if n:weighted+=n*mean/len(episodes)
        np.testing.assert_allclose(weighted,u.rollout_reward_mean,atol=2e-12,rtol=0)
        matched_episode_entries+=len(episodes)
    resume=json.loads((folder/'resume.json').read_text());assert resume['update']==32
    for slot,state in enumerate(resume['envs']):
        assert state['day']==final[slot]['day'] and state['k']==final[slot]['k']
        assert state['source']==(0 if group=='R' or slot<8 else 1)
        np.testing.assert_array_equal(state['completed'],np.bincount(sources[slots==slot],minlength=2))
        assert resume['controller']['k_rngs'][slot]==final[slot]['rng']
    if number%20==0:print(f'verified {number}/{len(jobs)} models',flush=True)

d=pd.DataFrame(points);assert len(d)==7680
d.to_csv(P/'actual_training_rewards_by_source.csv',index=False)
trend=[]
for (ev,group,source,seed),v in d.groupby(['event','group','source','seed'],sort=False):
    if v.n_episodes.sum()==0:continue
    v=v.sort_values('update');x=v.episode_reward_mean.to_numpy()
    trend.append({'event':ev,'group':group,'source':source,'seed':seed,
        'first4':float(x[:4].mean()),'last4':float(x[-4:].mean()),
        'gain':float(x[-4:].mean()-x[:4].mean())})
trend=pd.DataFrame(trend)
trend.to_csv(P/'actual_training_source_trends_per_seed.csv',index=False)
summary=trend.groupby(['event','group','source'],sort=False).agg(first4=('first4','mean'),last4=('last4','mean'),gain=('gain','mean'),seeds_improved=('gain',lambda x:int((x>0).sum()))).reset_index()
summary.to_csv(P/'actual_training_source_trends.csv',index=False)
assert d.query('group=="R" and source=="synthetic"').episode_reward_mean.isna().all()
assert d.query('group!="R"').n_episodes.gt(0).all()
audit={'status':'passed','models':len(jobs),'checkpoints':len(jobs)*32,
    'saved_episode_entries_verified':matched_episode_entries,'source_means_rows':len(d),
    'method':'Recover exact source from deterministic per-slot episode completion chronology, verified against every saved buffer and final RNG/env state.',
    'reward_definition':'actual stochastic episode r over k+21 days; classify by source of final21; synthetic source episodes include a real k-day prefix',
    'window':'source subsets of the same latest100 completed episodes used in Figure2, not100 per source',
    'n_epochs_or_rollouts_reexecuted':0,'all_source_weighted_means_reconstruct_figure2':True,
    'max_absolute_reconstruction_error':max_mixed_error,'R_synthetic':'not applicable; not replaced with zero',
    'checks':['every checkpoint num_timesteps/n_envs/n_steps','all saved episode k and length in exact slot completion order',
        'cumulative real/synthetic and k histogram counts','last_episode_starts','final env day/k/source/completion counts',
        'all16 final independent k RNG states','original mixed full/suffix reward','source-count weighted reconstruction']}
(P/'actual_training_source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(P/'actual_training_source_inputs.sha256.json').write_text(json.dumps(source_inputs,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2));print(summary.round(4).to_string(index=False))
