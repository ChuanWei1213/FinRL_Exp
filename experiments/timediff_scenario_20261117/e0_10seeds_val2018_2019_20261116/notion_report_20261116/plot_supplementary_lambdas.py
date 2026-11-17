"""Two supplementary generation figures; leave the lambda .5 main image intact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

P=Path(__file__).resolve().parent
R=Path('/home/cwei1213/TimeDiff/outputs/dow10/conditional/scenario_pilot')
SOURCES={.1:R/'run_20261114/generation',.5:R/'loss_ablation_20261115/lambda_0p5/generation',1.0:Path('/mnt/sda/cwei1213/timediff_finrl/lambda_extension_20261116/lambda_1p0/generation')}
EVENTS=['crash_2020','rebound_2020','bear_2022']
LABELS=['2020 急跌 E0','2020 反彈 E0','2022 熊市 E0']
DATA={(lam,ev):pd.read_csv(path/(ev+'_E0_f0')/'train/candidates.csv') for lam,path in SOURCES.items() for ev in EVENTS}
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#e2e8f0','savefig.facecolor':'white'})
for lam,name in [(.1,'06_lambda_0p1_generation_scatter'),(1.0,'07_lambda_1p0_generation_scatter')]:
 fig,axes=plt.subplots(1,3,figsize=(16,5.3))
 for ax,ev,label in zip(axes,EVENTS,LABELS):
  d=DATA[(lam,ev)];r=100*d.target_return.iloc[0];v=100*d.target_volatility.iloc[0]
  assert len(d)==4000 and d.loc[d.retained,'condition_hit'].all()
  ax.add_patch(Rectangle((r-.2*abs(r),v*.8),.4*abs(r),v*.4,
    edgecolor='#c26d17',facecolor='#fff2ce',alpha=.5,lw=1.8,zorder=1,label='兩軸各 ±20% 容許範圍'))
  ax.scatter(d.generated_return*100,d.generated_volatility*100,s=9,color='#94a3b8',alpha=.25,zorder=2,label='全部候選（4,000）')
  z=d[d.retained]
  ax.scatter(z.generated_return*100,z.generated_volatility*100,s=10,color='#147d92',alpha=.48,zorder=3,label='保留路徑')
  ax.scatter([r],[v],marker='*',s=210,color='#b91c1c',edgecolors='white',linewidths=.7,zorder=4,label='E0 目標')
  ax.set(title=f'{label}\n命中 {d.condition_hit.mean():.2%}｜保留 {len(z)}/800',xlabel='21 日市場報酬（%）',ylabel='年化波動率（%）')
  x=np.concatenate([DATA[(l,ev)].generated_return.to_numpy()*100 for l in SOURCES]+[np.array([r-.2*abs(r),r+.2*abs(r)])])
  y=np.concatenate([DATA[(l,ev)].generated_volatility.to_numpy()*100 for l in SOURCES]+[np.array([v*.8,v*1.2])])
  dx=np.ptp(x)*.05;dy=np.ptp(y)*.06
  ax.set_xlim(x.min()-dx,x.max()+dx);ax.set_ylim(max(0,y.min()-dy),y.max()+dy)
  ax.grid(alpha=.7);ax.set_axisbelow(True)
 h,l=axes[0].get_legend_handles_labels();order=[1,2,0,3]
 fig.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',bbox_to_anchor=(.5,-.035),ncol=4,frameon=False)
 fig.suptitle(f'TimeDiff λ={lam:.1f}｜三個 E0 的訓練候選',fontsize=17,y=1.035)
 fig.tight_layout()
 fig.savefig(P/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 print(name)
