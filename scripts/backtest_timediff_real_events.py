"""Frozen random-k PPO policies on actual 2020/2022 event prices.

Primary: first nominal 21 days, continuous buffered account with forecast timing.
Secondary: direct 21-day cash start and frozen-f0 full-event diagnostic.
Selection is locked from 2018 validation before computing event policy returns.
"""
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

STUDY=Path('/mnt/sda/cwei1213/timediff_finrl/random_k_20261116')
OUT=STUDY/'real_event_backtest_20261116'
GROUPS=['R','H','C','D','cash','equal_weight']
LABELS={'crash_2020':'2020 crash','rebound_2020':'2020 rebound','bear_2022':'2022 bear'}


def job_for(case,group):
    event=case.rsplit('_',2)[0]
    return f'{event}_{group}' if group in ('R','H') else f'{case}_{group}'+('_adaptive' if group=='D' else '')


def write_csv(path,frame):
    path=Path(path);temp=path.with_suffix('.tmp');frame.to_csv(temp,index=False);temp.replace(path)


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    assert json.loads((STUDY/'audit.json').read_text())['status']=='passed'
    torch.set_num_threads(2)
    forecasts=pd.read_csv(r.SOURCE/'forecasts.csv').set_index('forecast_id')
    generation=json.loads((r.SOURCE/'manifest.json').read_text())
    assert r.sha(generation['data'])==generation['data_sha256']
    result={'created_utc':r.now(),'seed':0,'generator_lambda':.5,'trained_k_max':21,
        'scope':'real-market diagnostic with existing frozen f0 policies; no new training/forecasts',
        'checkpoint_rule':'earliest argmax of 2018 real validation mean; final update32 sensitivity fixed in advance',
        'source_study':str(STUDY),'data':generation['data'],'data_sha256':generation['data_sha256'],
        'forecasts_sha256':r.sha(r.SOURCE/'forecasts.csv'),'script_sha256':r.sha(__file__),
        'fee':.0025,'initial_capital':1.,'initial_holdings':'all cash at each run start; first purchase charged',
        'primary':'fixed first21 dates within a continuous buffered account; H/C/D active on forecast dates, R otherwise',
        'full_event':'frozen f0 diagnostic without later 21-day forecast updates or continuation training',
        'aligned21':'cash start at nominal condition_start, one policy through21 days; timing error excluded in this diagnostic',
        'timing':'all H/C/D use the same case activation schedule; switch wealth/holdings without reset',
        'execution':'decision uses HLC ending previous close; charge rebalance then earn current close/previous close',
        'selection_source':'2018 only; event returns never select checkpoints, hyperparameters or dates',
        'formal_multiseed_experiment':False,'formal_event_holdout_now_inspected':True,
        'models':{},'cases':{}}
    for case in r.CASES:
        f=forecasts.loc[case].to_dict()
        result['cases'][case]=f
        for group in ('R','H','C','D'):
            name=job_for(case,group)
            if name in result['models']:continue
            folder=STUDY/'jobs'/name
            val=json.loads((folder/'validation_returns.json').read_text())
            means=[float(np.mean(z['real'])) for z in val]
            index=int(np.argmax(means))
            best=folder/('initial_model.zip' if index==0 else f'checkpoint_{index:03d}.zip')
            final=folder/'checkpoint_032.zip'
            a=PPO.load(best,device='cpu');b=PPO.load(folder/'best_model.zip',device='cpu')
            assert a.num_timesteps==index*r.BATCH and r.parameter_hash(a)==r.parameter_hash(b)
            result['models'][name]={'group':group,'validation_vector_sha256':r.sha(folder/'validation_returns.json'),
                'best2018':{'path':str(best),'sha256':r.sha(best),'update':index,'score':means[index],
                            'is_initialization':index==0,'parameter_sha256':r.parameter_hash(a)},
                'final32':{'path':str(final),'sha256':r.sha(final),'update':32,'score':means[32],
                           'is_initialization':False}}
    assert len(result['models'])==18
    r.write_json(manifest,result)
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    return result


class Market:
    def __init__(self,path):
        frame=pd.read_csv(path,parse_dates=['date'])
        pivots={key:frame.pivot(index='date',columns='tic',values=key)[r.TICKERS] for key in ('close','high','low')}
        self.dates=pivots['close'].index
        self.prices=np.stack([pivots[key].to_numpy() for key in ('close','high','low')],axis=-1)
        self.roundoff=r.original.price_roundoff(self.prices)
        assert self.dates.is_unique and self.dates.is_monotonic_increasing
        changes=pivots['close'].pct_change(fill_method=None)
        self.market_return=changes.mean(axis=1)
        self.previous5=(1+self.market_return).shift(1).rolling(5).apply(np.prod,raw=True)-1


def route(group,date,forecast,mode):
    active=forecast['activation_start']<=date<forecast['activation_end_exclusive']
    if group in ('cash','equal_weight'):return group,active
    if mode=='aligned21' or group=='R':return group,active
    return (group if active else 'R'),active


def run_account(market,models,forecast,group,mode):
    start=forecast['allowed_buffer_start'] if mode=='timed' else forecast['condition_start']
    end=forecast['allowed_buffer_end'] if mode=='timed' else forecast['condition_end']
    lo=int(market.dates.get_loc(pd.Timestamp(start)));hi=int(market.dates.get_loc(pd.Timestamp(end)))
    assert lo>=50
    wealth=1.;weights=np.r_[1.,np.zeros(10)];rows=[];previous_policy=None
    for index in range(lo,hi+1):
        date=market.dates[index].strftime('%Y-%m-%d')
        decision_date=market.dates[index-1].strftime('%Y-%m-%d')
        policy,active=route(group,date,forecast,mode)
        previous=weights.copy();before=wealth
        if policy=='cash':target=np.r_[1.,np.zeros(10)]
        elif policy=='equal_weight':target=np.r_[0.,np.ones(10)/10]
        else:
            obs=r.observations(market.prices[None],index-50,previous[None])
            with torch.inference_mode():action,_=models[policy].predict(obs,deterministic=True)
            target=r.action_weights(action)[0]
        gross=np.r_[1.,market.prices[index,:,0]/market.prices[index-1,:,0]]
        factor,weights,cost,turnover=r.transact(previous,target,gross)
        wealth*=float(factor)
        mu=1-float(cost);trades=mu*target[1:]-previous[1:]
        known_return=float(market.market_return.iloc[index-1])
        known_prev5=float(market.previous5.iloc[index-1])
        row={'date':date,'decision_date':decision_date,'group':group,'mode':mode,
            'active_policy':policy,'case_schedule_active':active,
            'policy_switch':bool(previous_policy is not None and previous_policy!=policy),
            'wealth_before':before,'wealth':wealth,'net_growth':float(factor),
            'reward':100*np.log(float(factor)),'fee_fraction':float(cost),'fee_wealth_units':before*float(cost),
            'turnover_fraction':float(turnover),'turnover_wealth_units':before*float(turnover),
            'stock_exposure_before':float(previous[1:].sum()),'target_stock_exposure':float(target[1:].sum()),
            'end_stock_exposure':float(weights[1:].sum()),
            'exposure_change_trade':float(target[1:].sum()-previous[1:].sum()),
            'exposure_change_price':float(weights[1:].sum()-target[1:].sum()),
            'known_market_return':known_return,'known_previous5_return':known_prev5,
            'rebound_signal':bool(known_return>0 and known_prev5<0),
            'pullback_signal':bool(known_return<0 and known_prev5>0)}
        for i,ticker in enumerate(r.TICKERS):
            row[f'previous_{ticker}']=float(previous[i+1]);row[f'target_{ticker}']=float(target[i+1])
            row[f'end_{ticker}']=float(weights[i+1]);row[f'trade_{ticker}']=float(before*trades[i])
        assert abs(sum(abs(x) for x in trades)-float(turnover))<1e-11
        assert abs(float(cost)-r.FEE*float(turnover))<1e-11
        assert np.isfinite(wealth) and wealth>0
        rows.append(row)
        previous_policy=policy
    return pd.DataFrame(rows)


def metrics(frame,start,end):
    data=frame[(frame.date>=start)&(frame.date<=end)]
    assert len(data)>0
    capital=float(data.wealth_before.iloc[0]);last=float(data.wealth.iloc[-1])
    path=np.r_[capital,data.wealth.to_numpy()]
    net=last/capital-1
    total_reward=100*np.log(last/capital)
    assert abs(total_reward-data.reward.sum())<1e-9
    return {'start':start,'end':end,'trading_days':len(data),'net_return':net,'log_reward':total_reward,
        'max_drawdown':float(np.min(path/np.maximum.accumulate(path)-1)),
        'turnover_sum':float(data.turnover_fraction.sum()),
        'cost_pct_start_capital':float(data.fee_wealth_units.sum()/capital),
        'mean_stock_exposure':float(data.end_stock_exposure.mean()),
        'mean_target_stock_exposure':float(data.target_stock_exposure.mean()),
        'starting_stock_exposure':float(data.stock_exposure_before.iloc[0]),
        'ending_stock_exposure':float(data.end_stock_exposure.iloc[-1]),
        'switch_count':int(data.policy_switch.sum())}


def behavior(frame,forecast,start,end):
    prediction=float(forecast['predicted_return'])
    signal='rebound_signal' if prediction<0 else 'pullback_signal'
    # Trigger is known at decision close d; rebalance earns next trading date d+1.
    mask=(frame.decision_date>=start)&(frame.decision_date<=end)&frame.case_schedule_active.astype(bool)
    opportunities=frame[mask&frame[signal].astype(bool)] if prediction!=0 else frame.iloc[:0]
    delta=opportunities.exposure_change_trade.to_numpy()
    effective=np.abs(delta)>.005+1e-12
    expected=(delta<-.005-1e-12) if prediction<0 else (delta>.005+1e-12)
    n=len(delta)
    return {'signal':'rebound_then_reduce' if prediction<0 else 'pullback_then_increase',
        'trigger_count':n,'expected_effective':int(expected.sum()),
        'opposite_effective':int((effective&~expected).sum()),'below_threshold':int((~effective).sum()),
        'mean_trade_exposure_change_pp':float(delta.mean()*100) if n else None,
        'expected_effective_rate':float(expected.mean()) if n else None,
        'sample_insufficient':n<5}


def worker(out,case,selection):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    manifest=json.loads((out/'manifest.json').read_text());forecast=manifest['cases'][case]
    folder=out/'runs'/f'{case}_{selection}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    market=Market(manifest['data']);models={}
    realized=market.market_return.loc[forecast['condition_start']:forecast['condition_end']]
    assert len(realized)==21 and abs((1+realized).prod()-1-float(forecast['actual_return']))<1e-12
    for group in ('R','H','C','D'):
        info=manifest['models'][job_for(case,group)][selection]
        assert r.sha(info['path'])==info['sha256']
        models[group]=PPO.load(info['path'],device='cpu')
        assert models[group].num_timesteps==info['update']*r.BATCH
    records=[];signals=[]
    for mode in ('timed','aligned21'):
        all_days=[]
        for group in GROUPS:
            daily=run_account(market,models,forecast,group,mode)
            daily['case']=case;daily['selection']=selection
            all_days.append(daily)
            periods={'first21':(forecast['condition_start'],forecast['condition_end'])}
            if mode=='timed':periods.update(full_event=(forecast['nominal_event_start'],forecast['nominal_event_end']),
                full_buffer=(forecast['allowed_buffer_start'],forecast['allowed_buffer_end']))
            for period,(start,end) in periods.items():
                record={'case':case,'event':forecast['event'],'setting':forecast['setting'],'selection':selection,
                    'mode':mode,'period':period,'group':group,**metrics(daily,start,end)}
                records.append(record)
                if mode=='timed':signals.append({**{k:record[k] for k in ['case','event','setting','selection','period','group']},
                    **behavior(daily,forecast,start,end)})
        write_csv(folder/f'daily_{mode}.csv',pd.concat(all_days,ignore_index=True))
    write_csv(folder/'metrics.csv',pd.DataFrame(records));write_csv(folder/'behavior.csv',pd.DataFrame(signals))
    r.write_json(folder/'complete.json',{'status':'complete','case':case,'selection':selection,
        'checkpoint_manifest_sha256':r.sha(out/'manifest.json'),'market_roundoff':market.roundoff,'completed_utc':r.now()})


def collect(out,manifest):
    summaries=[];signals=[];daily=[]
    for case in r.CASES:
        for selection in ('best2018','final32'):
            folder=out/'runs'/f'{case}_{selection}'
            state=json.loads((folder/'complete.json').read_text())
            assert state['status']=='complete' and state['checkpoint_manifest_sha256']==r.sha(out/'manifest.json')
            summaries.append(pd.read_csv(folder/'metrics.csv'))
            signals.append(pd.read_csv(folder/'behavior.csv'))
            for mode in ('timed','aligned21'):daily.append(pd.read_csv(folder/f'daily_{mode}.csv'))
    summary=pd.concat(summaries,ignore_index=True)
    behavior_rows=pd.concat(signals,ignore_index=True)
    days=pd.concat(daily,ignore_index=True)
    keys=['case','selection','mode','period']
    for control in ('R','H','equal_weight'):
        comparison=summary[summary['group']==control][keys+['net_return']].rename(columns={'net_return':f'control_{control}'})
        summary=summary.merge(comparison,on=keys,validate='many_to_one')
        summary[f'excess_{control}_pp']=100*(summary.net_return-summary[f'control_{control}'])
    write_csv(out/'performance.csv',summary)
    write_csv(out/'behavior.csv',behavior_rows)
    write_csv(out/'daily_trades.csv',days)
    checkpoint_rows=[]
    for case in r.CASES:
        for group in ('R','H','C','D'):
            name=job_for(case,group)
            for selection in ('best2018','final32'):
                info=manifest['models'][name][selection]
                checkpoint_rows.append({'case':case,'group':group,'selection':selection,'job':name,**info})
    write_csv(out/'checkpoint_selection.csv',pd.DataFrame(checkpoint_rows))
    return summary,behavior_rows,days


def plots(out,summary,days,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    colors={'R':'#666666','H':'#0072B2','C':'#E69F00','D':'#CC79A7','cash':'#222222','equal_weight':'#009E73'}
    styles={'R':'-','H':'--','C':'-.','D':'-','cash':':','equal_weight':'--'}
    labels={'R':'R','H':'H','C':'C','D':'D','cash':'Cash','equal_weight':'Daily equal stock'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    artifacts=[]
    specs=[('timed','first21','wealth','First 21 actual trading days | forecast timing applied'),
        ('timed','full_event','wealth','Full actual event | frozen f0 diagnostic, no forecast/model updates'),
        ('timed','full_buffer','wealth','Entire allowed timing buffer | continuous account'),
        ('aligned21','first21','wealth','Aligned 21-day cash-start diagnostic | timing error excluded'),
        ('timed','first21','exposure','First 21 actual trading days | stock allocation after rebalance')]
    with PdfPages(out/'real_event_charts.pdf') as pdf:
        for selection in ('best2018','final32'):
            for mode,period,measure,title in specs:
                fig,axes=plt.subplots(3,2,figsize=(14,11))
                for ax,case in zip(axes.flat,r.CASES):
                    f=manifest['cases'][case]
                    start,end=(f['condition_start'],f['condition_end']) if period=='first21' else (
                        (f['nominal_event_start'],f['nominal_event_end']) if period=='full_event' else
                        (f['allowed_buffer_start'],f['allowed_buffer_end']))
                    for group in GROUPS:
                        d=days[(days['case']==case)&(days['selection']==selection)&(days['mode']==mode)&
                            (days['group']==group)&(days.date>=start)&(days.date<=end)]
                        dates=pd.to_datetime(d.date)
                        if measure=='wealth':
                            x=pd.to_datetime([d.decision_date.iloc[0]]+d.date.tolist())
                            y=np.r_[100.,100*d.wealth.to_numpy()/d.wealth_before.iloc[0]]
                        else:x=dates;y=100*d.target_stock_exposure
                        ax.plot(x,y,label=labels[group],color=colors[group],ls=styles[group],lw=1.6)
                    if measure=='wealth':ax.axhline(100,color='#aaaaaa',lw=.6)
                    else:ax.set_ylim(-2,102)
                    if period=='full_buffer':
                        ax.axvspan(pd.Timestamp(f['nominal_event_start']),pd.Timestamp(f['nominal_event_end']),color='#777777',alpha=.06)
                        ax.axvline(pd.Timestamp(f['activation_start']),color='#0072B2',lw=.8,ls=':')
                        ax.axvline(pd.Timestamp(f['activation_end_exclusive']),color='#D55E00',lw=.8,ls=':')
                    locator=mdates.AutoDateLocator(minticks=3,maxticks=5)
                    ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    ax.set_title(f'{LABELS[f["event"]]} / {f["setting"]} | {start} to {end}',fontsize=10.5)
                    ax.set_ylabel('Net value (start of scored interval = 100)' if measure=='wealth' else 'Target stock allocation (%)')
                    ax.grid(axis='y',alpha=.2)
                handles,names=axes.flat[0].get_legend_handles_labels()
                fig.suptitle(title,fontsize=16,y=1.01)
                fig.text(.5,.965,'2018-selected checkpoints (including initialization if selected)' if selection=='best2018' else 'Final update32 checkpoints | predeclared sensitivity analysis',ha='center')
                fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=6,frameon=False)
                fig.text(.5,.008,'Dow10 + cash; long only; fee 0.0025. Seed0 exploratory event test. Subintervals retain holdings; complete buffer includes initial build cost.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.035,1,.9))
                name=f'{selection}_{mode}_{period}_{measure}.png'
                fig.savefig(out/name,dpi=170,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
                artifacts.append(name)
    return artifacts


def report(out,summary,signals,manifest,artifacts):
    lines=['# 隨機 k PPO：對應真實市場事件回測','',
        '使用 λ=0.5 合成資料、隨機 k=0–21 日訓練的既有 PPO；seed=0。這次直接使用真實價格，不重新訓練、不重新選預測、不按事件績效挑 checkpoint。', '',
        '## 選模、期間與交易方式','',
        '- 主表使用每個模型在 2018 真實驗證平均分數最高、同分取較早的 checkpoint；完整選模名單及雜湊在任何事件策略損益計算前寫入 manifest.json。最後 update32 另作預先指定的敏感度比較。',
        '- 部分 2018 最佳版本是 update0：這代表該組的訓練沒有超越初始化策略的驗證分數。主結果保留此規則，不因事件表現改選其他版本。',
        '- 主要期間是生成條件對應的固定首21日：急跌 2020-02-13～2020-03-13；反彈 2020-03-17～2020-04-15；熊市 2021-12-30～2022-01-28。已核對真實市場每日等權的無費用累積報酬與 forecasts.csv 的 actual_return 一致。',
        '- 時間誤差版本：帳戶從完整允許緩衝區的第一日全現金開始；H/C/D 僅在各 E0/E1 預測的啟用區間使用該組策略，其餘時間用該事件 R。換策略時資金與持倉連續；切換交易照常計費。R 始終使用 R；現金與每日等權基準維持各自規則。',
        '- 首21日與完整事件損益從該子區間開始前的實際帳戶狀態續算；不重新全現金建倉。提前啟用造成的區間前損益，反映在完整緩衝區的結果。完整緩衝區包含最初建倉成本。',
        '- 另提供 aligned21：在名義首日由全現金開始、整21日使用同一組模型，包含最初建倉成本；此項不施加時間誤差，用來檢查對應行情的直接表現。它與時間誤差版本也有起始持倉差異，差值不能全部歸因於時間誤差。',
        '- 完整實際事件為原定22／63／190日，但目前僅有首份 f0 模型與條件；完整事件表屬固定 f0 診斷，沒有每21日預測更新或續訓，不能當成完整分段實驗已完成。',
        '- 觀測只含截至前一交易日收盤的50日HLC與實際前次配置。費率0.0025，按自融資方程扣除股票買賣成本，然後獲得當日收盤／前收盤報酬；現金報酬為0。所有帳戶以1為初始淨值。', '',
        '## 使用的模型更新次數','',
        '| 條件 | R | H | C | D |','|---|---:|---:|---:|---:|']
    for case in r.CASES:
        f=manifest['cases'][case];updates=[manifest['models'][job_for(case,g)]['best2018']['update'] for g in ('R','H','C','D')]
        lines.append(f'| {LABELS[f["event"]]} / {f["setting"]} | '+' | '.join(str(x) for x in updates)+' |')
    for mode,period,title in [('timed','first21','主表：首21日，含時間誤差與連續帳戶'),
        ('aligned21','first21','對齊首21日：全現金起點、未施加時間誤差'),
        ('timed','full_event','完整事件：固定 f0 模型診斷'),
        ('timed','full_buffer','完整允許時間誤差緩衝區')]:
        lines+=['',f'## {title}','','數值為扣費後簡單報酬百分比；D−R／D−H 為百分點。','',
            '| 條件 | R | H | C | D | 每日等權 | 現金 | D−R | D−H |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for case in r.CASES:
            d=summary[(summary['case']==case)&(summary['selection']=='best2018')&(summary['mode']==mode)&(summary['period']==period)].set_index('group')
            f=manifest['cases'][case]
            values=[f'{100*d.loc[g,"net_return"]:+.2f}%' for g in GROUPS[:4]+['equal_weight','cash']]
            lines.append(f'| {LABELS[f["event"]]} / {f["setting"]} | '+' | '.join(values)+
                f' | {d.loc["D","excess_R_pp"]:+.2f} | {d.loc["D","excess_H_pp"]:+.2f} |')
    lines+=['','## 首21日：最大回撤、換手、成本與曝險','',
        '最大回撤用負百分比表示。換手為每日股票雙邊成交額／該日交易前淨值之累計，沒有除以2。成本以該評分子區間起始資金為分母；初始建倉若在子區間前發生，列在完整緩衝區成本。','',
        '| 條件／組別 | 最大回撤 | 累計換手 | 成本 | 平均股票曝險 |',
        '|---|---:|---:|---:|---:|']
    first=summary[(summary['selection']=='best2018')&(summary['mode']=='timed')&(summary['period']=='first21')]
    for case in r.CASES:
        f=manifest['cases'][case]
        for group in ('R','H','C','D'):
            a=first[(first['case']==case)&(first['group']==group)].iloc[0]
            lines.append(f'| {LABELS[f["event"]]} {f["setting"]} / {group} | {100*a.max_drawdown:.2f}% | {a.turnover_sum:.3f} | {100*a.cost_pct_start_capital:.3f}% | {100*a.mean_stock_exposure:.2f}% |')
    lines+=['','## 行為診斷','',
        '訊號只用決策收盤日 d 已知的市場報酬與 d 之前5日累積報酬，觀察其後一交易日 d+1 的實際再平衡。負預測看反彈後減碼，正預測看回撤後加碼。只計預測適用期內的匹配機會；R/H 與基準也用同一組機會日期。',
        '股票曝險變化使用「再平衡目標股票權重−再平衡前實際股票權重」，另外保存價格漂移造成的變化。超過0.5個百分點才算有效；機會少於5次標示小樣本，0次不計比率。這些描述性次數不能單獨證明已學會指定交易行為。','',
        '| 條件／組別 | 觸發 | 有效預期方向 | 有效反方向 | 未達門檻 | 平均交易曝險變化(pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for case in r.CASES:
        f=manifest['cases'][case]
        for group in ('R','H','C','D'):
            a=signals[(signals['case']==case)&(signals['selection']=='best2018')&(signals['period']=='first21')&(signals['group']==group)].iloc[0]
            change='—' if pd.isna(a.mean_trade_exposure_change_pp) else f'{a.mean_trade_exposure_change_pp:+.3f}'
            lines.append(f'| {LABELS[f["event"]]} {f["setting"]} / {group} | {int(a.trigger_count)} | {int(a.expected_effective)} | {int(a.opposite_effective)} | {int(a.below_threshold)} | {change} |')
    lines+=['','## 限制與檔案','',
        '- 這是使用事後實現值建立預測的情境機制實驗；本次只有seed0及每種事件一次市場實現。2020/2022事件現在已被檢視，不再宣稱是尚未查看的正式留出資料。沒有以本次結果重選模型、重抽預測或調整D。',
        '- `performance.csv` 包含2018最佳及final32的全部結果、相對R/H差異、最大回撤、成本及曝險。`daily_trades.csv` 包含逐日帳戶、啟用模型、所有股票前／目標／期末權重與成交金額（以初始資金1為單位）。',
        '- `behavior.csv` 保存各區間觸發、交易方向與小樣本標記；`checkpoint_selection.csv`、`manifest.json` 保存選模依據與凍結時間。`audit.json` 核對日期、帳戶連續性、費用、reward及輸入模型未改變。',
        '- `real_event_charts.pdf` 包含全部淨值與曝險圖；final32僅作預先指定的敏感度分析，未從兩套結果中按事件挑選較好版本。','']
    lines += [f'- [{name}]({out/name})' for name in artifacts]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def audit(out,manifest,summary,days,artifacts,started):
    assert len(summary)==288
    assert not summary.duplicated(['case','selection','mode','period','group']).any()
    assert np.isfinite(summary[['net_return','log_reward','max_drawdown','turnover_sum','cost_pct_start_capital']]).all().all()
    assert (summary[(summary['group']=='cash')].net_return==0).all()
    for key,d in days.groupby(['case','selection','mode','group'],sort=False):
        np.testing.assert_allclose(d.wealth_before.iloc[1:],d.wealth.iloc[:-1],rtol=0,atol=0)
        np.testing.assert_allclose(d.reward,100*np.log(d.wealth/d.wealth_before),atol=1e-10)
        np.testing.assert_allclose(d.fee_fraction,r.FEE*d.turnover_fraction,atol=1e-11)
        np.testing.assert_allclose(d.end_stock_exposure-d.stock_exposure_before,
            d.exposure_change_trade+d.exposure_change_price,atol=1e-11)
        assert (pd.to_datetime(d.decision_date)<pd.to_datetime(d.date)).all()
        if key[2]=='aligned21':assert len(d)==21
    for info in manifest['models'].values():
        for selection in ('best2018','final32'):assert r.sha(info[selection]['path'])==info[selection]['sha256']
    assert r.sha(manifest['data'])==manifest['data_sha256']
    assert r.sha(r.SOURCE/'forecasts.csv')==manifest['forecasts_sha256']
    assert r.sha(__file__)==manifest['script_sha256']
    result={'status':'passed','completed_utc':r.now(),'elapsed_seconds':time.monotonic()-started,
        'cases':6,'selections':2,'account_runs':144,'performance_rows':len(summary),'daily_rows':len(days),
        'models_unchanged':True,'2018_only_selection_frozen_before_policy_event_returns':True,
        'continuous_wealth_and_holdings':True,'reward_fee_and_drift_accounting':True,
        'new_training_steps':0,'segmented_forecast_updates_performed':False,
        'artifacts_sha256':{name:r.sha(out/name) for name in artifacts+['REPORT.md','real_event_charts.pdf',
            'performance.csv','behavior.csv','daily_trades.csv','checkpoint_selection.csv']}}
    r.write_json(out/'audit.json',result)
    r.write_json(out/'status.json',{**result,'status':'complete'})
    return result


def main(out,workers):
    started=time.monotonic()
    os.sched_setaffinity(0,list(range(9))+list(range(12,21)));os.nice(5)
    manifest=prepare(out)
    assert manifest['script_sha256']==r.sha(__file__)
    r.write_json(out/'status.json',{'status':'running','stage':'frozen_policy_real_event_rollouts','updated_utc':r.now()})
    tasks=[(case,selection) for case in r.CASES for selection in ('best2018','final32')]
    (out/'runs').mkdir(exist_ok=True)
    def one(case,selection):
        log=out/'runs'/f'{case}_{selection}.log'
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(out),'--case',case,'--selection',selection]
        with log.open('a') as handle:subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('Completed',case,selection,flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(one,*task) for task in tasks]
            for future in as_completed(futures):future.result()
        summary,signals,days=collect(out,manifest)
        artifacts=plots(out,summary,days,manifest)
        report(out,summary,signals,manifest,artifacts)
        result=audit(out,manifest,summary,days,artifacts,started)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        r.write_json(out/'status.json',{'status':'failed','error':str(error),'updated_utc':r.now()})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--case');p.add_argument('--selection',choices=['best2018','final32'])
    args=p.parse_args()
    if args.case:worker(args.out,args.case,args.selection)
    else:main(args.out,args.workers)
