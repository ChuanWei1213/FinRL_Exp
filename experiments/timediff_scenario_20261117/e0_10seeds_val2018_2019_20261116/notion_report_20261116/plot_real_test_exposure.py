"""Figure 5 supplement: actual end-of-day stock exposure on the same test paths."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

P=Path(__file__).resolve().parent
STUDY=P.parent
EVENTS=['crash_2020','rebound_2020','bear_2022']
LABELS=dict(zip(EVENTS,['2020 急跌','2020 反彈','2022 熊市']))
GROUPS=['R','H','C','D']
COLORS=dict(zip(GROUPS,['#64748b','#c17d11','#147d92','#8a4e9e']))
STYLES=dict(zip(GROUPS,['--',':','-','-.']))
TICKERS=['AAPL','AMGN','CRM','CSCO','IBM','INTC','MSFT','NKE','VZ','WMT']
daily=pd.read_csv(STUDY/'test_daily_trades.csv')
baseline=pd.read_csv(STUDY/'baseline_daily.csv')
test=pd.read_csv(STUDY/'test_per_seed.csv')
assert len(daily)==5040 and len(test)==240
keys=['selection','event','group','seed']
assert not daily.duplicated(keys+['date']).any()
assert set(daily.seed)==set(range(1,11))
assert np.isfinite(daily.end_stock_exposure).all()
assert daily.end_stock_exposure.between(-1e-12,1+1e-12).all()
np.testing.assert_allclose(daily[['end_'+t for t in TICKERS]].sum(axis=1),daily.end_stock_exposure,atol=1e-12,rtol=0)
averages=daily.groupby(keys).end_stock_exposure.mean().rename('recomputed').reset_index()
check=averages.merge(test[keys+['mean_stock_exposure']],on=keys,validate='one_to_one')
assert len(check)==240
np.testing.assert_allclose(check.recomputed,check.mean_stock_exposure,atol=1e-12,rtol=0)
first=daily.sort_values('date').groupby(keys).first()
np.testing.assert_allclose(first.stock_exposure_before,0,atol=1e-12,rtol=0)

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
 'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,
 'axes.spines.right':False,'grid.color':'#e2e8f0','grid.linewidth':.65,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(15.5,8.2),sharex=True,sharey=True)
curve_rows=[]
for row,sel in enumerate(['best_validation','final32']):
 for col,ev in enumerate(EVENTS):
  ax=axes[row,col];dates=None
  for gr in GROUPS:
   z=daily.query('selection==@sel and event==@ev and group==@gr').pivot(index='date',columns='seed',values='end_stock_exposure').sort_index()
   assert z.shape==(21,10) and z.notna().all().all()
   if dates is None:dates=z.index
   assert dates.equals(z.index)
   mean=np.r_[0,z.mean(axis=1).values];sd=np.r_[0,z.std(axis=1,ddof=1).values]
   ax.plot(range(22),mean,color=COLORS[gr],ls=STYLES[gr],lw=2,label=gr,zorder=3)
   ax.fill_between(range(22),mean-sd,mean+sd,color=COLORS[gr],alpha=.10,zorder=2)
   for day in range(22):
    curve_rows.append({'selection':sel,'event':ev,'group':gr,'trading_day':day,
       'date':'initial' if day==0 else str(z.index[day-1]),'mean_stock_exposure':mean[day],
       'sd_stock_exposure':sd[day],'n_seeds':10})
  for gr,target in [('equal_weight',1),('cash',0)]:
   b=baseline[(baseline['case']==ev+'_E0_f0')&(baseline['group']==gr)].sort_values('date')
   assert len(b)==21 and np.array_equal(b.date.to_numpy(),dates.to_numpy())
   np.testing.assert_allclose(b.end_stock_exposure,target,atol=1e-12,rtol=0)
   ax.plot(range(22),np.r_[0,b.end_stock_exposure.values],
      color='#172554' if gr=='equal_weight' else '#9ca3af',
      ls=(0,(1,4)) if gr=='equal_weight' else '-',lw=1.3 if gr=='equal_weight' else .8,
      label='每日等權' if gr=='equal_weight' else '全現金',zorder=1)
  ax.set_title(LABELS[ev]+('｜val 選模' if row==0 else '｜final32'))
  ax.set_ylabel('每日收盤股票曝險');ax.set_xlabel('事件內交易日')
  ax.set_ylim(-.035,1.055);ax.set_xlim(0,21)
  ax.set_yticks(np.linspace(0,1,6));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
  ax.set_xticks([0,5,10,15,21]);ax.tick_params(labelleft=True);ax.grid(alpha=.6)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.04))
fig.suptitle('圖 5 補充｜真實市場首 21 日的股票曝險比例',fontsize=17,y=1.015)
fig.tight_layout()
fig.savefig(P/'08_real_test_exposure.png',dpi=180,bbox_inches='tight');plt.close(fig)

fields=['selection','event','case','group','seed','date','stock_exposure_before',
 'target_stock_exposure','end_stock_exposure','exposure_change_trade','exposure_change_price']
export=daily[fields].copy()
export['end_cash_exposure']=1-export.end_stock_exposure
export.to_csv(P/'figure5_exposure_per_seed.csv',index=False)
curves=pd.DataFrame(curve_rows);curves.to_csv(P/'figure5_exposure_mean_sd.csv',index=False)
summary=daily.groupby(['selection','event','group']).end_stock_exposure.mean().unstack('group').reindex(columns=GROUPS)
summary.to_csv(P/'figure5_exposure_period_means.csv')
audit={'status':'passed','daily_rows':len(daily),'models':len(check),
 'actual_exposure_matches_sum_of_10_stock_weights':True,'period_means_match_test_metrics':True,
 'initial_exposure_all_cash':True,'date_windows_and_group_coverage_match_figure5':True,
 'seeds':list(range(1,11)),'metric':'end_stock_exposure; cash=1-stock',
 'standard_deviation':'sample SD across seeds, not a market confidence interval; band not clipped at 100%',
 'max_sd_band':float((curves.mean_stock_exposure+curves.sd_stock_exposure).max()),
 'sources_sha256':{name:hashlib.sha256((STUDY/name).read_bytes()).hexdigest() for name in ['test_daily_trades.csv','baseline_daily.csv','test_per_seed.csv']}}
(P/'figure5_exposure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(summary.round(4).to_string());print(json.dumps(audit,ensure_ascii=False))
