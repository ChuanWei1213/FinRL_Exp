"""Reproduce this report from frozen generation / main-job artifacts. No training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
STUDY = OUT.parent
GEN = Path('/home/cwei1213/TimeDiff/outputs/dow10/conditional/scenario_pilot/loss_ablation_20261115/lambda_0p5/generation')
EVENTS = ['crash_2020', 'rebound_2020', 'bear_2022']
LABELS = dict(zip(EVENTS, ['2020 急跌', '2020 反彈', '2022 熊市']))
GROUPS = ['R', 'H', 'C', 'D']
COLORS = dict(zip(GROUPS, ['#64748b', '#c17d11', '#147d92', '#8a4e9e']))
STYLES = dict(zip(GROUPS, ['--', ':', '-', '-.']))
SPLITS = ['train', 'validation', 'test']
inputs = set()

def csv(p):
    inputs.add(Path(p))
    return pd.read_csv(p)

def js(p):
    inputs.add(Path(p))
    return json.loads(Path(p).read_text())

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#e2e8f0', 'grid.linewidth': .65,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

figures = []
def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=180, bbox_inches='tight')
    figures.append((name, fig))

# Generation: three E0 only, retain original split labels for provenance.
forecasts = csv(GEN/'forecasts.csv')
prices=csv(Path('/home/cwei1213/TimeDiff/datasets/dow10/prices_2004.csv'))
close=prices.pivot(index='date',columns='tic',values='close').sort_index()
log_returns=np.log(close/close.shift(1))
event_corr={}
for ev in EVENTS:
    f=forecasts.loc[forecasts.forecast_id==ev+'_E0_f0'].iloc[0]
    x=log_returns.loc[f.condition_start:f.condition_end].to_numpy()
    assert x.shape==(21,10) and np.isfinite(x).all()
    event_corr[ev]=float(np.corrcoef(x.T)[np.triu_indices(10,1)].mean())
gen_rows = []
candidates = {}
for ev in EVENTS:
    fid = ev+'_E0_f0'
    for split in SPLITS:
        p = GEN/fid/split
        s = js(p/'summary.json')
        d = csv(p/'candidates.csv')
        assert len(d)==s['generated']
        assert int(d.retained.sum())==s['retained']
        assert int(d.condition_hit.sum())==s['condition_hits_before_selection']
        assert d.loc[d.retained, ['condition_hit','prices_valid']].all().all()
        assert s['retained']==s['requested'] and s['invalid_price_or_feature_count']==0
        target = forecasts.loc[forecasts.forecast_id==fid].iloc[0]
        assert np.allclose(d.target_return,target.predicted_return)
        assert np.allclose(d.target_volatility,target.predicted_volatility)
        expected = ((d.generated_return-target.predicted_return).abs()/abs(target.predicted_return)<=.2) & ((d.generated_volatility-target.predicted_volatility).abs()/abs(target.predicted_volatility)<=.2)
        assert np.array_equal(expected.to_numpy(), d.condition_hit.to_numpy())
        candidates[(ev,split)] = d
        dv=s['diversity']
        gen_rows.append(dict(event=ev,split=split,requested=s['requested'],generated=len(d),
            hits=int(d.condition_hit.sum()),hit_rate=s['condition_hit_rate_before_selection'],
            retained=s['retained'],retained_rate=s['retained_fraction_of_candidates'],
            valid_price_failures=s['invalid_price_or_feature_count'],
            correlation_generated=s['mean_pairwise_correlation_retained'],
            correlation_reference=s['mean_pairwise_correlation_paired_real'],
            correlation_actual_event=event_corr[ev],
            high_repair=s['high_repair_fraction_retained'],low_repair=s['low_repair_fraction_retained'],
            exact_duplicate_fraction=dv['exact_duplicate_fraction'],n_backgrounds=dv['n_unique_backgrounds'],
            max_per_background=dv['max_retained_per_background'],copied_future_count=dv['paired_real_future_copy_count_at_1e_minus_7']))
        gen_rows[-1].update(target_return=target.predicted_return,target_volatility=target.predicted_volatility,
            retained_mean_return=float(d.loc[d.retained,'generated_return'].mean()),
            retained_mean_volatility=float(d.loc[d.retained,'generated_volatility'].mean()))
generation=pd.DataFrame(gen_rows)
generation.to_csv(OUT/'generation_quality.csv',index=False)
candidate_export=[]
for (ev,split),d in candidates.items():
    z=d[['candidate_id','background_id','target_return','target_volatility','generated_return','generated_volatility','condition_hit','retained']].copy()
    z['event']=ev;z['split']=split;candidate_export.append(z)
pd.concat(candidate_export,ignore_index=True).to_csv(OUT/'generation_candidates.csv',index=False)

fig, axes = plt.subplots(1,3,figsize=(16,5.1))
for ax,ev in zip(axes,EVENTS):
    d=candidates[(ev,'train')]
    target=forecasts.loc[forecasts.forecast_id==ev+'_E0_f0'].iloc[0]
    r,v=100*target.predicted_return,100*target.predicted_volatility
    ax.scatter(d.generated_return*100,d.generated_volatility*100,s=9,color='#94a3b8',alpha=.22,rasterized=True,label='全部候選（4,000）')
    z=d[d.retained]
    ax.scatter(z.generated_return*100,z.generated_volatility*100,s=10,color='#147d92',alpha=.45,rasterized=True,label='保留（800）')
    ax.add_patch(Rectangle((r-.2*abs(r),v*.8),.4*abs(r),v*.4,edgecolor='#cf6f1a',facecolor='#fff3d6',alpha=.4,lw=2,zorder=2,label='生成容許區間（兩軸 ±20%）'))
    ax.scatter([r],[v],marker='*',s=220,color='#b91c1c',edgecolors='white',linewidths=.7,zorder=5,label='E0 目標')
    hit=generation.query('event==@ev and split=="train"').iloc[0].hit_rate
    ax.set(title=f'{LABELS[ev]}｜篩選前命中 {hit:.1%}',xlabel='21 日市場報酬（%）',ylabel='年化波動率（%）')
    ax.grid(alpha=.6);ax.set_axisbelow(True)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.04))
fig.suptitle('TimeDiff λ=0.5：條件能命中，但原始候選並非全部符合',fontsize=16,y=1.02)
fig.tight_layout();save(fig,'01_generation_scatter')

# Actual stochastic training reward, latest 100 completed episodes.
jobs=js(STUDY/'main_jobs.json'); assert len(jobs)==120
all_rollouts=[]; metrics=[]
for p in jobs:
    p=Path(p); j=js(p);assert j['phase']=='main' and j['seed'] in range(1,11)
    d=csv(p.parent/'updates.csv').sort_values('update')
    assert d['update'].tolist()==list(range(1,33))
    ev=j['case'].replace('_E0_f0','')
    d['event']=ev;d['group']=j['group'];d['seed']=j['seed']
    all_rollouts.append(d[['event','group','seed','update','steps','rollout_reward_mean','rollout_suffix_reward_mean']])
    x=d.rollout_reward_mean.to_numpy();z=d.rollout_suffix_reward_mean.to_numpy()
    metrics.append(dict(event=ev,group=j['group'],seed=j['seed'],
        first4=x[:4].mean(),last4=x[-4:].mean(),gain=x[-4:].mean()-x[:4].mean(),
        rising_transitions=int((np.diff(x)>0).sum()),transitions=31,
        suffix_gain=z[-4:].mean()-z[:4].mean()))
raw=pd.concat(all_rollouts,ignore_index=True); assert len(raw)==3840
assert not raw.isna().any().any()
raw.to_csv(OUT/'actual_training_curves.csv',index=False)
metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'training_trends_per_seed.csv',index=False)
trend=metrics.groupby(['event','group'],sort=False).agg(
    first4=('first4','mean'),last4=('last4','mean'),gain=('gain','mean'),
    seeds_improved=('gain',lambda x:int((x>0).sum())),
    rising_transitions=('rising_transitions','sum'),transitions=('transitions','sum'),
    suffix_gain=('suffix_gain','mean')).reset_index()
trend['rising_fraction']=trend.rising_transitions/trend.transitions
trend.to_csv(OUT/'training_trends.csv',index=False)
from plot_actual_training_reward import make_figure as make_actual_training_figure
fig,_=make_actual_training_figure(raw,trend)
save(fig,'02_actual_training_reward')

# Fixed deterministic train and real validation checks are separate measures.
lc=csv(STUDY/'learning_curves.csv'); assert len(lc)==3960
lc.to_csv(OUT/'fixed_evaluation_curves.csv',index=False)
fixed_metrics=[]
for (case,gr,seed),z in lc.groupby(['case','group','seed']):
    z=z.sort_values('update')
    fixed_metrics.append(dict(event=case.replace('_E0_f0',''),group=gr,seed=seed,
        train_gain=z.train_reward.iloc[-1]-z.train_reward.iloc[0],
        real_val_gain=z.val_real_reward.iloc[-1]-z.val_real_reward.iloc[0]))
fixed=pd.DataFrame(fixed_metrics).groupby(['event','group']).agg(train_gain=('train_gain','mean'),real_val_gain=('real_val_gain','mean')).reset_index()
fixed.to_csv(OUT/'fixed_eval_changes.csv',index=False)
fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
for row,ev in enumerate(EVENTS):
    for col,metric in enumerate(['train_reward','val_real_reward']):
        ax=axes[row,col]
        for gr in GROUPS:
            z=lc[(lc['case']==ev+'_E0_f0')&(lc['group']==gr)].groupby('steps')[metric].mean()
            ax.plot(z.index/1000,z.values,color=COLORS[gr],linestyle=STYLES[gr],lw=2,label=gr)
        ax.set_title(LABELS[ev]+('｜固定 train 評估' if col==0 else '｜2018–2019 真實 val'))
        ax.set_ylabel('21 日 suffix reward');ax.grid(alpha=.65)
        if row==2:ax.set_xlabel('環境互動步數（千）')
        ax.legend(ncol=4,frameon=False,fontsize=10)
fig.suptitle('固定評估：C／D 在訓練背景進步，急跌與熊市的真實 val 反而下降',fontsize=16,y=1.015)
fig.text(.5,-.015,'各線為 10 seeds 平均、確定性動作；train 為固定 64 真實／64 合成樣本（R 僅真實），val 僅真實。',ha='center',fontsize=10)
fig.tight_layout();save(fig,'03_fixed_train_and_real_val')

# Independently derive all paired effects from per-seed test records.
tests=csv(STUDY/'test_per_seed.csv');assert len(tests)==240
effects=[]
for (sel,ev),d in tests.groupby(['selection','event']):
    w=d.pivot(index='seed',columns='group',values='net_return')*100
    assert w.shape==(10,4) and w.notna().all().all()
    for a,b in [('H','R'),('C','R'),('C','H'),('D','R'),('D','H'),('D','C')]:
        delta=w[a]-w[b];mean=delta.mean();h=t.ppf(.975,9)*delta.std(ddof=1)/np.sqrt(10)
        effects.append(dict(selection=sel,event=ev,comparison=f'{a}-{b}',mean_delta_pp=mean,
            ci_low_pp=mean-h,ci_high_pp=mean+h,
            wins=int((delta>1e-8).sum()),ties=int((delta.abs()<=1e-8).sum()),losses=int((delta< -1e-8).sum())))
effects=pd.DataFrame(effects);effects.to_csv(OUT/'paired_effects.csv',index=False)
# Cross-check the existing paired file's values, not merely narrative text.
old=csv(STUDY/'paired_comparisons.csv')
old['event']=old['case'].str.replace('_E0_f0','',regex=False)
check=old.merge(effects,on=['selection','event','comparison'],validate='one_to_one',suffixes=('_old','_new'))
assert len(check)==30
assert np.allclose(check.mean_delta_pp_old,check.mean_delta_pp_new,atol=1e-10)
assert np.allclose(check.ci95_lower_pp,check.ci_low_pp,atol=1e-10)
assert np.allclose(check.ci95_upper_pp,check.ci_high_pp,atol=1e-10)

fig,axes=plt.subplots(1,2,figsize=(14,7.5),sharex=True,sharey=True)
comps=['C-R','C-H','D-C'];comp_colors={'C-R':'#147d92','C-H':'#c17d11','D-C':'#8a4e9e'}
labels=[]
for ev in EVENTS:
    for comp in comps:labels.append(f'{LABELS[ev]}　{comp}')
for ax,sel,title in zip(axes,['best_validation','final32'],['主結果：2018–2019 val 選模','敏感度：固定最後第 32 次更新']):
    for k,(ev,comp) in enumerate((e,c) for e in EVENTS for c in comps):
        a=effects.query('selection==@sel and event==@ev and comparison==@comp').iloc[0]
        ax.errorbar(a.mean_delta_pp,k,xerr=[[a.mean_delta_pp-a.ci_low_pp],[a.ci_high_pp-a.mean_delta_pp]],fmt='o',color=comp_colors[comp],capsize=4,lw=2,ms=6)
    ax.axvline(0,color='#475569',lw=1.2,ls='--');ax.set_title(title)
    ax.set_yticks(range(9),labels);ax.set_ylim(8.6,-.6);ax.set_xlabel('同 seed 扣費後報酬差（百分點）')
    ax.grid(axis='x',alpha=.65);ax.set_xlim(-5,14)
    for y in [2.5,5.5]:ax.axhline(y,color='#cbd5e1',lw=.8)
axes[1].tick_params(labelleft=True)
fig.suptitle('條件資料的效益取決於選模方式；D 相對 C 未呈現一致額外增益',fontsize=16,y=1.02)
fig.text(.5,-.015,'點：10 組配對 seed 的平均；橫線：95% t 區間（df=9，未校正多重比較）。C−H 比較指定條件與歷史條件。',ha='center',fontsize=10)
fig.tight_layout();save(fig,'04_paired_test_effects')

daily=csv(STUDY/'test_daily_trades.csv');base_daily=csv(STUDY/'baseline_daily.csv')
daily[['seed','event','case','group','selection','date','wealth']].to_csv(OUT/'real_test_wealth.csv',index=False)
base_daily[['case','group','date','wealth']].to_csv(OUT/'baseline_wealth.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(15.5,8.2),sharex=True,sharey='col')
for row,sel in enumerate(['best_validation','final32']):
    for col,ev in enumerate(EVENTS):
        ax=axes[row,col]
        for gr in GROUPS:
            z=daily.query('selection==@sel and event==@ev and group==@gr').pivot(index='date',columns='seed',values='wealth').sort_index()
            assert z.shape==(21,10)
            m=np.r_[1,z.mean(axis=1).values];sd=np.r_[0,z.std(axis=1).values]
            ax.plot(range(22),m,color=COLORS[gr],ls=STYLES[gr],lw=2,label=gr)
            ax.fill_between(range(22),m-sd,m+sd,color=COLORS[gr],alpha=.08)
        b=base_daily[(base_daily['case']==ev+'_E0_f0')&(base_daily['group']=='equal_weight')].sort_values('date')
        ax.plot(range(22),np.r_[1,b.wealth.values],color='#172554',ls=(0,(1,4)),lw=1.4,label='每日等權')
        ax.axhline(1,color='#9ca3af',lw=.8,label='全現金')
        ax.set_title(LABELS[ev]+('｜val 選模' if row==0 else '｜final32'))
        ax.set_ylabel('扣費後淨值（初始＝1）');ax.set_xlabel('事件內交易日');ax.grid(alpha=.6)
        ax.set_xticks([0,5,10,15,21])
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.04))
fig.suptitle('真實市場首 21 日：主選模的差異較小，final32 的條件組差異較大',fontsize=16,y=1.015)
fig.tight_layout();save(fig,'05_real_test_wealth')

# Exact summary tables and reproducibility manifest.
summary=tests.groupby(['selection','event','group']).agg(mean_return=('net_return','mean'),sd_return=('net_return','std'),mean_dd=('max_drawdown','mean'),mean_exposure=('mean_stock_exposure','mean'),mean_cost=('cost_pct_start_capital','mean')).reset_index()
summary.to_csv(OUT/'test_returns.csv',index=False)
selected=csv(STUDY/'selected_checkpoints.csv')
initial=selected.groupby(['case','group']).initialization_selected.sum().unstack('group').reindex(columns=GROUPS)
initial.to_csv(OUT/'initialization_selected.csv')
conditions=forecasts[forecasts.setting.eq('E0')][['forecast_id','event','condition_start','condition_end','condition_days','predicted_return','predicted_volatility']]
conditions.to_csv(OUT/'conditions.csv',index=False)

with PdfPages(OUT/'report_figures.pdf') as pdf:
    for name,fig in figures:pdf.savefig(fig,bbox_inches='tight')
for _,fig in figures:plt.close(fig)

manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
(OUT/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
info=dict(main_jobs=len(jobs),rollout_rows=len(raw),test_rows=len(tests),
    selected_initializations=int(selected.initialization_selected.sum()),
    improved_training_seeds=int((metrics.gain>0).sum()),
    generation_candidates=int(generation.generated.sum()),generation_retained=int(generation.retained.sum()),
    weighted_generation_hit_rate=float(generation.hits.sum()/generation.generated.sum()),
    single_seed_upward_step_fraction_min=float(trend.rising_fraction.min()),
    single_seed_upward_step_fraction_max=float(trend.rising_fraction.max()),
    sources=len(manifest))
(OUT/'analysis_summary.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
print('TRAIN TRENDS\n',trend.round(4).to_string(index=False))
print('PAIRED C-H\n',effects[effects.comparison.eq('C-H')].round(6).to_string(index=False))
