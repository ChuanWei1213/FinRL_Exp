"""2019 diagnostic evaluation and research figures after 2018-only D calibration."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from finrl.experiments import timediff_random_k as r

OLD=Path('/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115')
GROUPS=['R','H','C','D']
LABELS={'crash_2020':'2020 crash','rebound_2020':'2020 rebound','bear_2022':'2022 bear'}


def job_for(case,group):
    event=case.rsplit('_',2)[0]
    return f'{event}_{group}' if group in ('R','H') else f'{case}_{group}'+('_adaptive' if group=='D' else '')


def label(case):
    event,error,_=case.rsplit('_',2)
    return f'{LABELS[event]} / {error}'


def post_worker(out,name):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    job=out/'jobs'/name;c=json.loads((job/'job.json').read_text())
    dest=out/'diagnostics';dest.mkdir(exist_ok=True)
    path=dest/f'{name}.json'
    prior=json.loads(path.read_text()) if path.exists() else {'test':[]}
    real=np.load(out/'banks/real_test.npy')
    fid='H_historical' if c['group']=='H' else c['case']
    syn=np.load(out/'banks'/f'{fid}_test.npy') if c['group']!='R' else None
    for update in range(len(prior['test']),33):
        file=job/('initial_model.zip' if update==0 else f'checkpoint_{update:03d}.zip')
        model=PPO.load(file,device='cpu');assert model.num_timesteps==update*r.BATCH
        a=r.real_evaluation(model,real)
        b=r.evaluate(model,syn,r.fixed_ks(len(syn)),True) if syn is not None else {key:np.array([]) for key in a}
        prior['test'].append({'update':update,'real':a['suffix'],'real_full':a['full'],
            'synthetic':b['suffix'],'synthetic_full':b['full'],'checkpoint_sha256':r.sha(file)})
        r.write_json(path,prior)
    # The same fixed random-k evaluation protocol is applied to the legacy k=0 policy.
    if 'legacy_k0' not in prior:
        file=OLD/'jobs'/name/'latest_model.zip'
        old=PPO.load(file,device='cpu')
        val=np.load(out/'banks/real_validation.npy')
        prior['legacy_k0']={}
        for split,prices in [('validation',val),('test',real)]:
            values=r.real_evaluation(old,prices)
            prior['legacy_k0'][split]={key:float(x.mean()) for key,x in values.items()}
        prior['legacy_k0']['checkpoint_sha256']=r.sha(file)
    prior['status']='complete';prior['completed_utc']=r.now()
    r.write_json(path,prior)


def baseline(prices,ks,target):
    wealth=np.ones(len(prices));prefix=np.ones(len(prices));weights=np.zeros((len(prices),11));weights[:,0]=1
    ks=np.asarray(ks)
    for day in range(int(ks.max())+21):
        active=np.flatnonzero(day<ks+21);boundary=active[ks[active]==day];prefix[boundary]=wealth[boundary]
        indices=71-ks[active]+day
        gross=np.column_stack([np.ones(len(active)),prices[active,indices,:,0]/prices[active,indices-1,:,0]])
        factor,w,_cost,_turnover=r.transact(weights[active],np.broadcast_to(target,(len(active),11)),gross)
        wealth[active]*=factor;weights[active]=w
    return {'full':100*np.log(wealth),'suffix':100*np.log(wealth/prefix)}


def summarize(out):
    rows=[];raw=[];legacy=[];jobs=[]
    for case in r.CASES:
        for group in GROUPS:
            name=job_for(case,group);folder=out/'jobs'/name
            val=json.loads((folder/'validation_returns.json').read_text())
            post=json.loads((out/'diagnostics'/f'{name}.json').read_text())
            updates=pd.read_csv(folder/'updates.csv')
            if name not in jobs:jobs.append(name)
            for v,t in zip(val,post['test']):
                assert v['update']==t['update']
                for split,record,realkey,synkey in [('train',v,'train_real','train_synthetic'),
                    ('validation',v,'real','synthetic'),('test',t,'real','synthetic')]:
                    for suffix in ('','_full'):
                        real=np.array(record[realkey+suffix]);syn=np.array(record[synkey+suffix])
                        rows.append({'case':case,'group':group,'job':name,'update':v['update'],
                            'steps':v['update']*r.BATCH,'split':split,'metric':'suffix_21d' if not suffix else 'full_episode',
                            'real_n':len(real),'synthetic_n':len(syn),'real_reward':real.mean(),
                            'synthetic_reward':syn.mean() if len(syn) else None,
                            'matched_reward':.5*(real.mean()+syn.mean()) if len(syn) else real.mean()})
            for row in updates.to_dict('records'):
                raw.append({**row,'case':case,'group':group,'job':name})
            for split in ('validation','test'):
                new=next(z for z in rows[::-1] if z['case']==case and z['group']==group and z['split']==split and z['metric']=='suffix_21d' and z['update']==32)
                old=post['legacy_k0'][split]['suffix']
                legacy.append({'case':case,'group':group,'split':split,'legacy_k0_reward':old,
                    'random_k_reward':new['real_reward'],'random_minus_legacy':new['real_reward']-old})
    curves=pd.DataFrame(rows);rollout=pd.DataFrame(raw)
    curves.to_csv(out/'reward_curves.csv',index=False);rollout.to_csv(out/'rollout_rewards.csv',index=False)
    pd.DataFrame(legacy).to_csv(out/'legacy_k0_common_evaluation.csv',index=False)
    records=[];initials=set();inventory={}
    for status in sorted((out/'jobs').glob('*/status.json')):
        state=json.loads(status.read_text());assert state['status']=='complete'
        config=state['config'];directory=status.parent;u=pd.read_csv(directory/'updates.csv')
        initials.add(state['initialization']['policy_sha256'])
        assert state['steps']==config['updates']*r.BATCH
        assert config['source_code_sha256']==r.sha(r.__file__)
        assert (u.actual_real_price_steps+u.actual_synthetic_price_steps==u.steps).all()
        if config['group']!='R':assert (u.completed_real_episodes==u.completed_synthetic_episodes).all()
        for checkpoint in ['initial_model.zip']+[f'checkpoint_{i:03d}.zip' for i in range(1,config['updates']+1)]:
            inventory[str(directory/checkpoint)]=r.sha(directory/checkpoint)
        records.append({'name':directory.name,'group':config['group'],'updates':config['updates'],'steps':state['steps'],
            'last_validation_score':state['last_validation_score'],'best_validation_score':state['best_validation_score'],
            'training_seconds':u.training_seconds.sum(),'validation_seconds':u.validation_seconds.sum(),
            'real_episodes':int(u.completed_real_episodes.iloc[-1]),'synthetic_episodes':int(u.completed_synthetic_episodes.iloc[-1]),
            'synthetic_price_steps':int(u.actual_synthetic_price_steps.iloc[-1]),
            'mean_k':u.mean_completed_k.iloc[-1],'pool_tail_steps':int(u.stale_pool_continuation_steps.iloc[-1])})
    assert len(initials)==1
    legacy_init=json.loads((OLD/'jobs/crash_2020_R/initialization.json').read_text())['policy_sha256']
    assert next(iter(initials))==legacy_init,'Architecture/initialization must also match the legacy experiment'
    for case in r.CASES:
        c=json.loads((out/'jobs'/job_for(case,'C')/'updates.json').read_text())
        d=json.loads((out/'jobs'/job_for(case,'D')/'updates.json').read_text())
        for a,b in zip(c,d):
            for key in ('steps','completed_k_counts','prefix_steps','suffix_steps',
                        'completed_real_episodes','completed_synthetic_episodes'):
                assert a[key]==b[key],(case,key)
    r.write_json(out/'checkpoint_inventory.json',inventory)
    pd.DataFrame(records).to_csv(out/'job_summary.csv',index=False)
    baselines=[]
    for split in ('validation','test'):
        prices=np.load(out/'banks'/f'real_{split}.npy')
        for name,target in [('cash',np.r_[1.,np.zeros(10)]),('daily_equal_stock',np.r_[0.,np.ones(10)/10])]:
            values=baseline(np.repeat(prices,22,axis=0),np.tile(np.arange(22),len(prices)),target)
            baselines.append({'split':split,'baseline':name,**{k:float(v.mean()) for k,v in values.items()}})
    pd.DataFrame(baselines).to_csv(out/'baselines.csv',index=False)
    return curves,rollout,pd.DataFrame(records),initials.pop()


def figures(out,curves,rollout):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'train':'#0072B2','validation':'#D55E00','test':'#009E73'}
    groupcolors={'R':'#666666','H':'#0072B2','C':'#E69F00','D':'#CC79A7'}
    dashes={'R':'-','H':'--','C':'-.','D':':'}
    artifacts=[]
    def style(ax):
        ax.axhline(0,color='#999999',lw=.7);ax.grid(axis='y',alpha=.2)
        ax.set_xlim(0,32*r.BATCH);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:.0f}k'))
        ax.set_xlabel('PPO interactions');ax.set_ylabel('Mean 21-day suffix reward')
    with PdfPages(out/'all_reward_curves.pdf') as pdf:
        def save(fig,name):
            fig.savefig(out/name,dpi=170,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig);artifacts.append(name)
        for measure,prefix,description in [('matched_reward','','R real only; H/C/D 50% real + 50% synthetic episodes'),
            ('real_reward','common_real_','Identical real targets for all groups; validation/test averaged over all 22 k values')]:
            for group in GROUPS:
                cases=[f'{e}_E0_f0' for e in r.EVENTS] if group in ('R','H') else r.CASES
                nrows,ncols=(1,3) if len(cases)==3 else (3,2)
                fig,axes=plt.subplots(nrows,ncols,figsize=(16,6) if nrows==1 else (13.5,11),squeeze=False)
                for ax,case in zip(axes.flat,cases):
                    for split in ('train','validation','test'):
                        d=curves[(curves['case']==case)&(curves['group']==group)&(curves['split']==split)&(curves['metric']=='suffix_21d')]
                        ax.plot(d.steps,d[measure],label={'train':'Train (fixed sample)','validation':'Val (2018)','test':'Test (2019)'}[split],
                            color=colors[split],marker='o',ms=2.4,lw=1.6)
                    ax.set_title(label(case) if group in ('C','D') else LABELS[case.rsplit('_',2)[0]]+' / E0,E1 shared')
                    style(ax)
                handles,labels=axes.flat[0].get_legend_handles_labels()
                fig.suptitle(f'{group}: random k=0..21 | train / validation / test',fontsize=17,y=1.01)
                fig.text(.5,.94 if nrows==1 else .965,description,ha='center')
                fig.legend(handles,labels,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.895 if nrows==1 else .94))
                fig.text(.5,.008,'Reward of final 21 traded days after a real prelude; wealth/holdings carry through. Fee 0.0025. Seed 0; diagnostic test.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.035,1,.84 if nrows==1 else .9))
                save(fig,f'{prefix}{group}_train_val_test.png')
        for metric,name,title in [('rollout_reward_mean','actual_training_full_reward.png','Full episode: k real days + 21 real/synthetic days'),
            ('rollout_suffix_reward_mean','actual_training_suffix_reward.png','Final 21 traded days, after a real prelude')]:
            fig,axes=plt.subplots(3,2,figsize=(13.5,11))
            for ax,case in zip(axes.flat,r.CASES):
                for group in GROUPS:
                    d=rollout[(rollout['case']==case)&(rollout['group']==group)]
                    ax.plot(d.steps,d[metric],label=group,color=groupcolors[group],ls=dashes[group],marker='o',ms=2.4)
                style(ax);ax.set_title(label(case));ax.set_ylabel('Mean completed-episode reward')
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.suptitle('Actual stochastic training reward | R / H / C / D',fontsize=17,y=1.01)
            fig.text(.5,.965,title,ha='center');fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
            fig.text(.5,.01,'Latest 100 completed training episodes collected before that PPO update; raw values without smoothing.',ha='center')
            fig.tight_layout(rect=(0,.04,1,.9));save(fig,name)
        fig,axes=plt.subplots(3,2,figsize=(13.5,10))
        for ax,case in zip(axes.flat,r.CASES):
            data=rollout[(rollout['case']==case)&(rollout['group']=='D')]
            ax.bar(data['update'],data.replay_gain,color=['#009E73' if x>0 else '#D55E00' for x in data.replay_gain])
            ax.axhline(0,color='#777777',lw=.8);ax.set_title(label(case));ax.set_xlabel('PPO update')
            ax.set_ylabel('Same-pool suffix reward gain');ax.grid(axis='y',alpha=.2)
        fig.suptitle('D: paired training-pool gains with fixed diagnostic k per path',fontsize=16,y=1.01)
        fig.tight_layout();save(fig,'D_same_pool_update_gains.png')
        comparison=pd.read_csv(out/'legacy_k0_common_evaluation.csv')
        fig,ax=plt.subplots(figsize=(12,5.5));x=np.arange(len(r.CASES))
        for shift,split in [(-.18,'validation'),(.18,'test')]:
            values=[comparison[(comparison['case']==case)&(comparison['group']=='D')&(comparison['split']==split)].iloc[0].random_minus_legacy for case in r.CASES]
            ax.bar(x+shift,values,.34,label='Val 2018' if split=='validation' else 'Test 2019',color=colors[split])
        ax.axhline(0,color='#777777',lw=.8);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(x,[label(case).replace(' / ','\n') for case in r.CASES])
        ax.set_ylabel('Final policy reward: random k - legacy k=0')
        ax.set_title('D: both policies evaluated on identical real targets and all 22 k values')
        ax.legend(frameon=False);fig.text(.5,.01,'21-day suffix after a traded real prelude. Seed 0; legacy training pools/calibration differ, so this is descriptive.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1));save(fig,'D_legacy_k0_comparison.png')
    return artifacts


def report(out,curves,rollout,summary,artifacts):
    settings=json.loads((out/'D_selected_parameters.json').read_text())
    bank=json.loads((out/'banks/manifest.json').read_text())
    test=json.loads((out/'banks/test_manifest.json').read_text())
    plan=json.loads((out/'plan.json').read_text())
    wall_minutes=(pd.Timestamp(r.now())-pd.Timestamp(plan['created_utc'])).total_seconds()/60
    lines=['# 隨機前置 k=0–21 日：PPO 重訓與 D 重新校準','',
        '本輪使用既有 TimeDiff λ=0.5 合成路徑，從隨機初始化重新訓練 PPO，僅 seed=0 開發前測。舊 k=0 結果保留原狀；沒有執行正式 2020/2022 真實事件回測。','',
        '## 設定與公平比較','',
        '- 每次 reset 抽整數 k∈{0,…,21}。50 日觀測歷史 → k 日真實交易 → 完整 21 日真實／合成交易。前置日正常計費與計算 reward；進入合成段不重設資金、持倉或觀測窗口。',
        '- PPO 仍使用相同 EIIE、50 日 HLC 與前次配置、初始全現金、費率 0.0025、每日 100×log 扣費後資產比。預測／來源旗標／k 不進入 observation。',
        '- 每更新 16×504=8,064 次實際互動；主要模型各 32 更新=258,048 次。跨 rollout 的未完成 episode 自然延續。相同 seed 起始權重一致、C/D 相同互動預算與價格路徑庫。',
        '- 真實／合成配對環境共用 k 抽樣序列，以維持精確 50% 的來源 episode 與來源環境互動數；不同配對使用不同 RNG。k 與路徑抽樣 RNG 分開。因合成來源 episode 的前置 k 日也是真實行情，實際合成交易日比例約為總互動的三分之一，逐工作另報實際數量。',
        '- D 每次更新重新收集 rollout。達換池條件後，已開始的 episode 沿原路徑走完，最多 41 個交易步後使用新池；不把未完成交易截斷。實際延續舊池的步數與換池理由逐更新保存。',
        '- 使用 18/24 邏輯 CPU 的 affinity 上限、8 個程序×每程序 2 個 torch 執行緒、較低排程優先度；現有 CUDA13 與驅動不相容，未使用 GPU，也未改動驅動或系統套件。','',
        '## 資料、驗證及指標','',
        '- 50+k 日所需的額外歷史由原始真實行情補齊。先要求所有 k 都能合法使用該路徑，再抽 k；沒有為了可用路徑而重抽較小 k。',
        f'- 真實訓練 {bank["banks"]["real_train"]["n"]} 條，真實 2018 驗證 {bank["banks"]["real_validation"]["n"]} 個非重疊目標；真實 2019 診斷測試 {test["banks"]["real_test"]["n"]} 個。歷史與目標均位於各自資料分割內。',
        '- 既有合成保留路徑中，少數位於分割起點、無法提供 71 日歷史者排除；原始 ID 與排除數均在銀行 manifest。未重新生成或更換預測目標，品質門檻不變。',
        '- 全 episode reward 含 k+21 日，另外保存最後 21 日 suffix reward。suffix 從前置日結束時的實際資金／持倉續算，包含當日交易成本；不是在合成起點重新建倉。',
        '- 2018 選模與 D 校準主指標為最後 21 日的純 reward。真實驗證對每個目標平均全部 22 個 k，再以原目標為樣本單位計算標準誤，不把 k 視為獨立行情樣本。合成驗證／測試各路徑使用固定、平衡分配的 k；抽樣不依績效。',
        '- train 曲線為固定 64 條真實／64 條合成路徑的確定性評估；val/test 為各合法保留路徑。H/C/D 的混合曲線為兩來源平均 reward 各佔 50%。另提供各組相同真實資料的曲線，以及原始隨機探索訓練曲線。',
        '- 2019 只在 D 設定決定後進行診斷評估，不參與選模或調參。2019 先前已被檢視，不能再宣稱為全新未見測試。所有情境標籤仍屬事後機制實驗。','',
        '## D 重新校準','',
        f'重複更新數比較 1/2/4/8；依 2018 配對改善決定是否測 32/64 池。此次選擇 pool={settings["environment_pool_size"]}、最短 {settings["minimum_repeats"]}、最長 {settings["maximum_repeats"]} 次更新。',
        f'改善門檻 {settings["improvement_threshold"]:.6f}，退步門檻 {settings["regression_threshold"]:.6f}。仍為單種子前測結果；所有未改善與較差候選保留。','',
        '## D 訓練與最終真實資料表現','',
        '| 條件 | 原始全episode train：前4→後4 | 原始suffix train：前4→後4 | 真實Val Δ初始 | 真實Test Δ初始 | 最終Val D−C | 最終Test D−C |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in r.CASES:
        u=rollout[(rollout['case']==case)&(rollout['group']=='D')]
        values=[]
        for split in ('validation','test'):
            d=curves[(curves['case']==case)&(curves['group']=='D')&(curves['split']==split)&(curves['metric']=='suffix_21d')]
            c=curves[(curves['case']==case)&(curves['group']=='C')&(curves['split']==split)&(curves['metric']=='suffix_21d')]
            values.append((d.iloc[-1].real_reward-d.iloc[0].real_reward,d.iloc[-1].real_reward-c.iloc[-1].real_reward))
        lines.append(f'| {label(case)} | {u.rollout_reward_mean.head(4).mean():.3f}→{u.rollout_reward_mean.tail(4).mean():.3f} | '
            f'{u.rollout_suffix_reward_mean.head(4).mean():.3f}→{u.rollout_suffix_reward_mean.tail(4).mean():.3f} | '
            f'{values[0][0]:+.3f} | {values[1][0]:+.3f} | {values[0][1]:+.3f} | {values[1][1]:+.3f} |')
    lines+=['','## 與舊 k=0 模型對照及限制','',
        '`legacy_k0_common_evaluation.csv` 將兩輪最終模型放在本輪相同的真實目標、全部 22 個 k 上評估，避免直接比較 21 日和 k+21 日總 reward。舊模型的訓練可用路徑與 D 校準不同，因此這不是只改 k 的完整因果消融；結果仍是單種子描述性比較。',
        '隨機 k 排除固定起點的一項捷徑，並不保證生成多樣性、真實泛化或指定增減碼行為。基準現金與每日等權的扣費後分數另存 baselines.csv。','',
        '## 產出','',f'完成 {len(summary)} 個工作，共 {int(summary.steps.sum()):,} 次訓練互動。自 tmux 啟動至本報告產生約 {wall_minutes:.1f} 分鐘（不含環境修改與事前測試）。',
        '- `reward_curves.csv`、`rollout_rewards.csv`：所有曲線數值與 k、交易日來源、池延續紀錄。',
        '- `D_calibration_candidates.csv`、`D_selected_parameters.json`：本輪 D 校準。',
        '- `job_summary.csv`、`diagnostics/*.json`、`checkpoint_inventory.json`：逐工作／逐 checkpoint 結果與雜湊。',
        '- `environment_tests.log`、`integration_smoke/audit.json`、`audit.json`：環境測試、跨 episode 續跑與結果完整性核對。',
        '- `all_reward_curves.pdf`：完整圖表。','']
    lines += [f'- [{name}]({out/name})' for name in artifacts]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(out,workers):
    started=time.monotonic()
    assert (out/'training_complete.json').exists()
    r.build_banks(out,test=True)
    jobs=[]
    for case in r.CASES:
        for group in GROUPS:
            name=job_for(case,group)
            if name not in jobs:jobs.append(name)
    (out/'diagnostics').mkdir(exist_ok=True)
    def one(name):
        with (out/'diagnostics'/f'{name}.log').open('a') as log:
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(out),'--job',name]
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Diagnostic complete',name,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(one,name) for name in jobs]
        for future in as_completed(futures):future.result()
    curves,rollout,summary,initial=summarize(out)
    artifacts=figures(out,curves,rollout)
    report(out,curves,rollout,summary,artifacts)
    assert len(curves)==6*4*33*3*2 and np.isfinite(curves[['real_reward','matched_reward']]).all().all()
    for file in ['manifest.json','test_manifest.json']:
        manifest=json.loads((out/'banks'/file).read_text())
        for name,meta in manifest['banks'].items():assert r.sha(out/'banks'/f'{name}.npy')==meta['sha256']
    assert json.loads((out/'integration_smoke/audit.json').read_text())['status']=='passed'
    old_config=json.loads((OLD/'jobs/crash_2020_R/job.json').read_text())
    assert r.sha(r.original.__file__)==old_config['source_code_sha256']
    assert r.sha(r.__file__)==json.loads((out/'integration_smoke/audit.json').read_text())['module_sha256']
    result={'status':'passed','completed_utc':r.now(),'jobs':len(summary),'primary_jobs':len(jobs),
        'total_training_interactions':int(summary.steps.sum()),'paired_initial_policy_sha256':initial,
        'source_module_sha256':r.sha(r.__file__),'finalizer_sha256':r.sha(__file__),
        'unchanged_original_helpers_sha256':r.sha(r.original.__file__),
        'fixed_258048_interactions_per_primary_model':True,'natural_episode_boundaries':True,
        'mid_episode_resume_bit_identical':True,'selection_used_2018_only':True,
        'C_D_k_streams_and_source_episode_counts_match':True,'initialization_matches_legacy_k0':True,
        'actual_2020_2022_event_test':False,'postprocessing_seconds':time.monotonic()-started,
        'artifacts_sha256':{name:r.sha(out/name) for name in artifacts+['REPORT.md','all_reward_curves.pdf','reward_curves.csv',
            'rollout_rewards.csv','job_summary.csv','D_selected_parameters.json','legacy_k0_common_evaluation.csv']}}
    r.write_json(out/'audit.json',result)
    r.write_json(out/'status.json',{**result,'status':'complete'})
    (out/'finalizer_source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--job')
    args=p.parse_args()
    if args.job:post_worker(args.out,args.job)
    else:main(args.out,args.workers)
