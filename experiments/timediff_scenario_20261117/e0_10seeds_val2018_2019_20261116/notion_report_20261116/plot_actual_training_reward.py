"""Render Figure 2 with an independent y range for every scenario/group."""
from pathlib import Path
import hashlib,json,shutil
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

def make_figure(raw,trend):
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
        'font.size':11,'axes.titlesize':13,'axes.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#e2e8f0',
        'grid.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True,sharey=False)
    ranges=[]
    for row,ev in enumerate(EVENTS):
        for col,gr in enumerate(GROUPS):
            ax=axes[row,col];d=raw.query('event==@ev and group==@gr')
            assert len(d)==320 and d.seed.nunique()==10 and d.rollout_reward_mean.notna().all()
            for _,z in d.groupby('seed'):
                z=z.sort_values('steps')
                ax.plot(z.steps/1000,z.rollout_reward_mean,color=COLORS[gr],alpha=.15,lw=.65)
            z=d.groupby('steps').rollout_reward_mean.agg(['mean','std'])
            lower=z['mean']-z['std'];upper=z['mean']+z['std']
            ax.fill_between(z.index/1000,lower,upper,color=COLORS[gr],alpha=.12)
            ax.plot(z.index/1000,z['mean'],color=COLORS[gr],lw=2.2)
            lo=min(float(d.rollout_reward_mean.min()),float(lower.min()))
            hi=max(float(d.rollout_reward_mean.max()),float(upper.max()))
            pad=max((hi-lo)*.08,1e-6)
            ax.set_ylim(lo-pad,hi+pad)
            if lo-pad<=0<=hi+pad:
                ax.axhline(0,color='#94a3b8',lw=.7)
            a=trend.query('event==@ev and group==@gr').iloc[0]
            ax.set_title(f'{LABELS[ev]} / {gr}  Δ={a.gain:+.2f}',fontsize=12)
            ax.grid(alpha=.55);ax.set_xlim(8,259)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.tick_params(axis='y',labelleft=True)
            if col==0:ax.set_ylabel('訓練 episode reward')
            if row==2:ax.set_xlabel('環境互動步數（千）')
            ranges.append({'event':ev,'group':gr,'data_min':lo,'data_max':hi,
                           'y_lower':lo-pad,'y_upper':hi+pad})
    fig.suptitle('實際 train reward：後期平均上升，各次更新仍反覆波動',fontsize=17,y=1.01)
    fig.text(.5,-.015,'每個子圖採獨立 y 軸。粗線：10 seeds 平均；陰影：±1 seed SD；細線：單一 seed。\nΔ＝更新 29–32 平均 − 更新 1–4 平均。',ha='center',fontsize=11)
    fig.tight_layout()
    return fig,ranges

if __name__=='__main__':
    sources=[P/'actual_training_curves.csv',P/'training_trends.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    original=P/'02_actual_training_reward.png'
    previous=P/'02_actual_training_reward_shared_y_backup.png'
    if original.exists() and not previous.exists():shutil.copy2(original,previous)
    raw=pd.read_csv(sources[0]);trend=pd.read_csv(sources[1])
    fig,ranges=make_figure(raw,trend)
    fig.savefig(original,dpi=180,bbox_inches='tight')
    fig.savefig(P/'02_actual_training_reward_independent_y.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert len(set((r['y_lower'],r['y_upper']) for r in ranges))==12
    (P/'figure2_independent_y_audit.json').write_text(json.dumps({'status':'passed',
        'subplot_count':12,'sharey':False,'all_seed_values_and_sd_bands_within_limits':True,
        'source_data_unchanged':True,'sources_sha256':hashes,'axis_ranges':ranges},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'subplots':12,'independent_y_ranges':len(ranges),'data_unchanged':True}))
