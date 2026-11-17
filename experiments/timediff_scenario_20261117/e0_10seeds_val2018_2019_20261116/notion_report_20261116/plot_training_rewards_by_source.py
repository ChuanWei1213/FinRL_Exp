"""Two source-separated stochastic rollout figures aligned with Figure 2."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator,ScalarFormatter

P=Path(__file__).resolve().parent
EVENTS=['crash_2020','rebound_2020','bear_2022']
LABELS=dict(zip(EVENTS,['2020 急跌','2020 反彈','2022 熊市']))
GROUPS=['R','H','C','D']
COLORS=dict(zip(GROUPS,['#64748b','#c17d11','#147d92','#8a4e9e']))
data=pd.read_csv(P/'actual_training_rewards_by_source.csv')
trend=pd.read_csv(P/'actual_training_source_trends.csv')
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
 'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,
 'axes.spines.right':False,'grid.color':'#e2e8f0','grid.linewidth':.65,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
ranges=[]
for source,label,filename in [('synthetic','合成來源','09_actual_training_synthetic_reward'),('real','真實來源','10_actual_training_real_reward')]:
    fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True,sharey=False)
    for row,ev in enumerate(EVENTS):
        for col,gr in enumerate(GROUPS):
            ax=axes[row,col];d=data.query('source==@source and event==@ev and group==@gr')
            assert len(d)==320
            if source=='synthetic' and gr=='R':
                assert d.n_episodes.eq(0).all() and d.episode_reward_mean.isna().all()
                ax.set_axis_off();ax.text(.5,.55,f'{LABELS[ev]} / R',ha='center',va='center',transform=ax.transAxes,fontsize=12,color='#64748b')
                ax.text(.5,.4,'無合成來源訓練 episode',ha='center',va='center',transform=ax.transAxes,color='#64748b')
                continue
            assert d.episode_reward_mean.notna().all() and d.n_episodes.gt(0).all() and d.seed.nunique()==10
            for _,z in d.groupby('seed'):
                z=z.sort_values('steps');ax.plot(z.steps/1000,z.episode_reward_mean,color=COLORS[gr],alpha=.15,lw=.65)
            z=d.groupby('steps').episode_reward_mean.agg(['mean','std'])
            low=z['mean']-z['std'];high=z['mean']+z['std']
            ax.fill_between(z.index/1000,low,high,color=COLORS[gr],alpha=.12)
            ax.plot(z.index/1000,z['mean'],color=COLORS[gr],lw=2.2)
            lo=min(float(d.episode_reward_mean.min()),float(low.min()));hi=max(float(d.episode_reward_mean.max()),float(high.max()))
            pad=max((hi-lo)*.08,1e-6);ax.set_ylim(lo-pad,hi+pad)
            if lo-pad<=0<=hi+pad:ax.axhline(0,color='#94a3b8',lw=.7)
            t=trend.query('source==@source and event==@ev and group==@gr').iloc[0]
            ax.set_title(f'{LABELS[ev]} / {gr}  Δ={t.gain:+.2f}',fontsize=12)
            ax.set_xlim(8,259);ax.grid(alpha=.55)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if col==0 or (source=='synthetic' and col==1):ax.set_ylabel('訓練 episode reward')
            if row==2:ax.set_xlabel('環境互動步數（千）')
            ranges.append({'source':source,'event':ev,'group':gr,'lower':lo-pad,'upper':hi+pad})
    fig.suptitle(f'圖 2 補充｜實際 train reward：{label} episode',fontsize=17,y=1.01)
    fig.text(.5,-.015,'同圖 2 最近 100 個完成 episode，按末 21 日來源拆分；reward 涵蓋 k＋21 日。每格獨立 y 軸。\n粗線：10 seeds 平均；陰影：±1 seed SD；細線：單一 seed。Δ＝末四次更新平均 − 前四次更新平均。',ha='center',fontsize=11)
    fig.tight_layout();fig.savefig(P/(filename+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    print(filename,flush=True)
(P/'training_source_plot_audit.json').write_text(json.dumps({'status':'passed',
 'images':2,'real_subplots':12,'synthetic_subplots':9,'R_synthetic_not_applicable_panels':3,
 'sharey':False,'source':'actual stochastic rollout episode rewards, reconstructed and audited from saved checkpoint buffers',
 'all_seed_values_and_sd_bands_in_range':True,'axis_ranges':ranges,
 'source_csv_sha256':hashlib.sha256((P/'actual_training_rewards_by_source.csv').read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+'\n')
