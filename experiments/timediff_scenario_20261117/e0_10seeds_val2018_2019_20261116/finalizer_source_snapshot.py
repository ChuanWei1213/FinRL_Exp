"""Lock validation-selected policies, then test the3 actual E0 first21 windows."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from finrl.experiments import timediff_e0_multiseed as m
spec=importlib.util.spec_from_file_location('frozen_event_engine',ROOT/'scripts/backtest_timediff_real_events.py')
bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)

GROUPS=['R','H','C','D']
SELECTIONS=['best_validation','final32']
LABELS={'crash_2020':'2020 crash','rebound_2020':'2020 rebound','bear_2022':'2022 bear'}
T95_DF9=2.262157162854099


def job_for(case,group,seed):
    event=case.rsplit('_',2)[0]
    logical=f'{event}_{group}' if group in ('R','H') else f'{case}_{group}'
    return f'{logical}_seed{seed:02d}'


def lock_selection(out):
    target=out/'test_selection_locked.json'
    if target.exists():return json.loads(target.read_text())
    torch.set_num_threads(2)
    forecasts=pd.read_csv(m.SOURCE/'forecasts.csv').set_index('forecast_id')
    banks=json.loads((out/'banks/manifest.json').read_text())
    payload={'created_utc':m.now(),'main_seeds':m.SEEDS,'cases':{},'models':{},
        'rule':'earliest argmax of2018-2019 real validation21-day suffix mean; final32 sensitivity',
        'test':'actual first21 trading days, deterministic, cash start, initial purchase charged',
        'test_evaluation_started':False,'train_module_sha256':m.sha(m.__file__),
        'bank_manifest_sha256':m.sha(out/'banks/manifest.json'),
        'data':banks['price_source'],'data_sha256':banks['price_source_sha256'],
        'forecasts_sha256':m.sha(m.SOURCE/'forecasts.csv'),
        'backtest_engine_sha256':m.sha(ROOT/'scripts/backtest_timediff_real_events.py')}
    for case in m.CASES:
        f=forecasts.loc[case].to_dict();assert f['setting']=='E0' and f['condition_days']==21
        payload['cases'][case]=f
    for seed in m.SEEDS:
        for case in m.CASES:
            for group in GROUPS:
                name=job_for(case,group,seed);folder=out/'jobs'/name
                state=json.loads((folder/'status.json').read_text())
                assert state['status']=='complete' and state['config']['seed']==seed and state['config']['phase']=='main'
                rows=json.loads((folder/'validation_returns.json').read_text())
                assert [row['update'] for row in rows]==list(range(33))
                scores=[float(np.mean(row['real'])) for row in rows];best=int(np.argmax(scores))
                selected=folder/('initial_model.zip' if best==0 else f'checkpoint_{best:03d}.zip')
                a=PPO.load(selected,device='cpu');b=PPO.load(folder/'best_model.zip',device='cpu')
                assert a.seed==seed and a.num_timesteps==best*m.BATCH and m.parameter_hash(a)==m.parameter_hash(b)
                payload['models'][name]={'seed':seed,'case':case,'group':group,
                    'validation_returns_sha256':m.sha(folder/'validation_returns.json'),
                    'best_validation':{'path':str(selected),'sha256':m.sha(selected),'update':best,
                        'validation_score':scores[best],'initialization_selected':best==0},
                    'final32':{'path':str(folder/'checkpoint_032.zip'),'sha256':m.sha(folder/'checkpoint_032.zip'),
                        'update':32,'validation_score':scores[32],'initialization_selected':False}}
    assert len(payload['models'])==120
    m.write_json(target,payload)
    return payload


def test_worker(out,seed):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    manifest=json.loads((out/'test_selection_locked.json').read_text())
    folder=out/'event_test'/f'seed{seed:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    market=bt.Market(manifest['data']);rows=[];daily=[];signals=[];errors=[]
    for case in m.CASES:
        f=manifest['cases'][case]
        actual=market.market_return.loc[f['condition_start']:f['condition_end']]
        assert len(actual)==21 and abs((1+actual).prod()-1-f['actual_return'])<1e-12
        index=market.dates.get_loc(pd.Timestamp(f['condition_start']))
        array=market.prices[None,index-71:index+21]
        for selection in SELECTIONS:
            for group in GROUPS:
                info=manifest['models'][job_for(case,group,seed)][selection]
                assert m.sha(info['path'])==info['sha256']
                model=PPO.load(info['path'],device='cpu')
                assert model.seed==seed and model.num_timesteps==info['update']*m.BATCH
                frame=bt.run_account(market,{group:model},f,group,'aligned21')
                record=bt.metrics(frame,f['condition_start'],f['condition_end'])
                check=float(m.evaluate(model,array,np.array([0]))[0])
                errors.append(abs(check-record['log_reward']));assert errors[-1]<1e-6
                tags={'seed':seed,'case':case,'event':f['event'],'group':group,'selection':selection,'selected_update':info['update']}
                rows.append({**tags,**record})
                signals.append({**tags,**bt.behavior(frame,f,f['condition_start'],f['condition_end'])})
                for key,value in tags.items():frame[key]=value
                daily.append(frame)
    bt.write_csv(folder/'performance.csv',pd.DataFrame(rows))
    bt.write_csv(folder/'behavior.csv',pd.DataFrame(signals))
    bt.write_csv(folder/'daily.csv',pd.concat(daily,ignore_index=True))
    m.write_json(folder/'complete.json',{'status':'complete','seed':seed,'selection_manifest_sha256':m.sha(out/'test_selection_locked.json'),
        'max_vectorized_crosscheck_error':max(errors),'completed_utc':m.now()})


def collect(out,manifest):
    perf=[];days=[];signals=[]
    for seed in m.SEEDS:
        folder=out/'event_test'/f'seed{seed:02d}'
        assert json.loads((folder/'complete.json').read_text())['status']=='complete'
        perf.append(pd.read_csv(folder/'performance.csv'));days.append(pd.read_csv(folder/'daily.csv'));signals.append(pd.read_csv(folder/'behavior.csv'))
    perf=pd.concat(perf,ignore_index=True);days=pd.concat(days,ignore_index=True);signals=pd.concat(signals,ignore_index=True)
    assert len(perf)==240 and len(days)==5040
    bt.write_csv(out/'test_per_seed.csv',perf);bt.write_csv(out/'test_daily_trades.csv',days);bt.write_csv(out/'test_behavior.csv',signals)
    summaries=[];pairs=[]
    for (selection,case,group),frame in perf.groupby(['selection','case','group'],sort=False):
        assert sorted(frame.seed.tolist())==m.SEEDS
        values=frame.net_return.to_numpy()*100;mean=float(values.mean());std=float(values.std(ddof=1));half=T95_DF9*std/np.sqrt(10)
        summaries.append({'selection':selection,'case':case,'group':group,'n':10,'mean_return_pct':mean,'std_return_pct':std,
            'ci95_lower_pct':mean-half,'ci95_upper_pct':mean+half,
            'mean_max_drawdown_pct':float(frame.max_drawdown.mean()*100),
            'mean_cost_pct':float(frame.cost_pct_start_capital.mean()*100),'mean_turnover':float(frame.turnover_sum.mean()),
            'mean_stock_exposure_pct':float(frame.mean_stock_exposure.mean()*100),
            'initialization_selected_count':int((frame.selected_update==0).sum())})
    summary=pd.DataFrame(summaries);bt.write_csv(out/'test_summary.csv',summary)
    for selection in SELECTIONS:
        for case in m.CASES:
            matrix=perf[(perf.selection==selection)&(perf['case']==case)].pivot(index='seed',columns='group',values='net_return').loc[m.SEEDS]
            for a,b in [('H','R'),('C','R'),('D','R'),('D','H'),('D','C')]:
                delta=(matrix[a]-matrix[b]).to_numpy()*100
                mean=float(delta.mean());std=float(delta.std(ddof=1));half=T95_DF9*std/np.sqrt(10)
                pairs.append({'selection':selection,'case':case,'comparison':f'{a}-{b}','n':10,
                    'mean_delta_pp':mean,'std_delta_pp':std,'ci95_lower_pp':mean-half,'ci95_upper_pp':mean+half,
                    'wins':int((delta>1e-10).sum()),'ties':int((np.abs(delta)<=1e-10).sum()),'losses':int((delta< -1e-10).sum())})
    pairs=pd.DataFrame(pairs);bt.write_csv(out/'paired_comparisons.csv',pairs)
    market=bt.Market(manifest['data']);base=[];base_days=[]
    for case,f in manifest['cases'].items():
        for group in ('cash','equal_weight'):
            frame=bt.run_account(market,{},f,group,'aligned21')
            base.append({'case':case,'group':group,**bt.metrics(frame,f['condition_start'],f['condition_end'])})
            frame['case']=case;base_days.append(frame)
    baselines=pd.DataFrame(base);bt.write_csv(out/'baselines.csv',baselines)
    bt.write_csv(out/'baseline_daily.csv',pd.concat(base_days,ignore_index=True))
    learning=[];jobs=[];initials={};inventory={}
    for jobpath in json.loads((out/'main_jobs.json').read_text()):
        folder=Path(jobpath).parent;config=json.loads(Path(jobpath).read_text());state=json.loads((folder/'status.json').read_text())
        assert state['status']=='complete' and config['phase']=='main' and config['seed'] in m.SEEDS
        assert config['source_code_sha256']==m.sha(m.__file__)
        initials.setdefault(config['seed'],set()).add(state['initialization']['policy_sha256'])
        case=f'{config["case"]}_E0_f0' if config['group'] in ('R','H') else config['case']
        val=json.loads((folder/'validation_returns.json').read_text());updates=pd.read_csv(folder/'updates.csv')
        for row in val:
            real=np.mean(row['real']);syn=np.mean(row['synthetic']) if row['synthetic'] else None
            tr=np.mean(row['train_real']);ts=np.mean(row['train_synthetic']) if row['train_synthetic'] else None
            learning.append({'seed':config['seed'],'case':case,'group':config['group'],'update':row['update'],'steps':row['update']*m.BATCH,
                'train_real_reward':tr,'train_synthetic_reward':ts,'train_reward':tr if ts is None else .5*(tr+ts),
                'val_real_reward':real,'val_synthetic_reward':syn,'val_reward':real if syn is None else .5*(real+syn)})
        jobs.append({'job':folder.name,'seed':config['seed'],'case':case,'group':config['group'],'steps':state['steps'],
            'training_seconds':updates.training_seconds.sum(),'validation_seconds':updates.validation_seconds.sum(),
            'real_episodes':int(updates.completed_real_episodes.iloc[-1]),'synthetic_episodes':int(updates.completed_synthetic_episodes.iloc[-1]),
            'mean_k':float(updates.mean_completed_k.iloc[-1]),'actual_synthetic_price_fraction':float(updates.actual_synthetic_price_steps.iloc[-1]/state['steps'])})
        for file in ['initial_model.zip','best_model.zip','latest_model.zip']+[f'checkpoint_{i:03d}.zip' for i in range(1,33)]:inventory[str(folder/file)]=m.sha(folder/file)
    assert all(len(values)==1 for values in initials.values()) and len({next(iter(v)) for v in initials.values()})==10
    learning=pd.DataFrame(learning);bt.write_csv(out/'learning_curves.csv',learning)
    bt.write_csv(out/'main_job_summary.csv',pd.DataFrame(jobs));m.write_json(out/'checkpoint_inventory.json',inventory)
    selected=[{'job':name,'seed':info['seed'],'case':info['case'],'group':info['group'],**info['best_validation']} for name,info in manifest['models'].items()]
    bt.write_csv(out/'selected_checkpoints.csv',pd.DataFrame(selected))
    return perf,days,signals,summary,pairs,baselines,learning


def figures(out,days,summary,learning,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'R':'#666666','H':'#0072B2','C':'#E69F00','D':'#CC79A7','cash':'#222222','equal_weight':'#009E73'}
    artifacts=[]
    baseline=pd.read_csv(out/'baseline_daily.csv')
    with PdfPages(out/'charts.pdf') as pdf:
        def save(fig,name):
            fig.savefig(out/name,dpi=160,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig);artifacts.append(name)
        for group in GROUPS:
            for realonly,prefix in [(False,''),(True,'real_only_')]:
                fig,axes=plt.subplots(1,3,figsize=(16,6))
                for ax,case in zip(axes,m.CASES):
                    data=learning[(learning['case']==case)&(learning['group']==group)]
                    fields=[('train_real_reward' if realonly else 'train_reward','#0072B2','Train'),
                            ('val_real_reward' if realonly else 'val_reward','#D55E00','Val2018-2019')]
                    for key,color,label in fields:
                        a=data.groupby('steps')[key].agg(['mean','std']).reset_index()
                        ax.plot(a.steps/1000,a['mean'],color=color,label=label,lw=1.8)
                        ax.fill_between(a.steps/1000,a['mean']-a['std'],a['mean']+a['std'],color=color,alpha=.14)
                    ax.axhline(0,color='#aaaaaa',lw=.7);ax.grid(axis='y',alpha=.2)
                    ax.set_title(LABELS[case.rsplit('_',2)[0]]);ax.set_xlabel('Training interactions (thousands)');ax.set_ylabel('Mean21-day suffix reward')
                handles,labels=axes[0].get_legend_handles_labels()
                fig.suptitle(f'{group}: train / validation learning curves | PPO seeds1-10',fontsize=17,y=1.01)
                fig.text(.5,.94,'Real data only' if realonly else 'R real only; H/C/D50% real +50% synthetic evaluation means',ha='center')
                fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.895),ncol=2,frameon=False)
                fig.text(.5,.008,'Lines: seed mean. Bands: one seed SD. Fixed evaluation paths/k; no event-test checkpoint search.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.035,1,.84));save(fig,f'{prefix}{group}_learning_curves.png')
        for selection in SELECTIONS:
            for measure in ['wealth','exposure']:
                fig,axes=plt.subplots(1,3,figsize=(16,6))
                for ax,case in zip(axes,m.CASES):
                    f=manifest['cases'][case]
                    for group in GROUPS:
                        d=days[(days['case']==case)&(days['selection']==selection)&(days['group']==group)]
                        field='wealth' if measure=='wealth' else 'target_stock_exposure'
                        a=d.groupby('date')[field].agg(['mean','std']).reset_index();dates=pd.to_datetime(a.date)
                        ax.plot(dates,a['mean']*100,color=colors[group],label=group,lw=1.8)
                        ax.fill_between(dates,(a['mean']-a['std'])*100,(a['mean']+a['std'])*100,color=colors[group],alpha=.12)
                    if measure=='wealth':
                        for group in ['cash','equal_weight']:
                            d=baseline[(baseline['case']==case)&(baseline['group']==group)]
                            ax.plot(pd.to_datetime(d.date),d.wealth*100,color=colors[group],ls='--',label='Cash' if group=='cash' else 'Daily equal stock')
                        ax.axhline(100,color='#aaaaaa',lw=.6)
                    else:ax.set_ylim(-2,102)
                    ax.set_title(f'{LABELS[f["event"]]}\n{f["condition_start"]} to {f["condition_end"]}',fontsize=10.5)
                    ax.set_ylabel('Net account value (initial cash=100)' if measure=='wealth' else 'Stock allocation after rebalance (%)')
                    locator=mdates.AutoDateLocator(minticks=3,maxticks=4)
                    ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator));ax.grid(axis='y',alpha=.2)
                handles,labels=axes[0].get_legend_handles_labels()
                fig.suptitle('Actual E0 first21 trading days | PPO seeds1-10',fontsize=17,y=1.01)
                fig.text(.5,.94,'Checkpoints selected on2018-2019 validation' if selection=='best_validation' else 'Final update32: predeclared sensitivity analysis',ha='center')
                fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.895),ncol=6,frameon=False)
                fig.text(.5,.008,'Lines: seed mean; bands: one seed SD. Cash start; fee0.0025 including first purchase. No forecast/model update during test.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.035,1,.84));save(fig,f'{selection}_test_{measure}.png')
    return artifacts


def report(out,summary,pairs,baselines,signals,manifest,artifacts):
    plan=json.loads((out/'plan.json').read_text());bank=json.loads((out/'banks/manifest.json').read_text())
    settings=json.loads((out/'D_parameters_locked.json').read_text())
    wall=(pd.Timestamp(m.now())-pd.Timestamp(plan['created_utc'])).total_seconds()/60
    lines=['# TimeDiff λ=0.5 × PPO：三個E0、10 seeds','',
        '使用既有λ=0.5合成資料，只保留急跌、反彈、熊市的E0。PPO seeds1–10；seed0僅作D校準及開發，不納入10種子結果。所有模型重新隨機初始化，各情境獨立訓練。','',
        '## 資料與設定','',
        '- Train：沿用上一輪完全相同的2005–2017價格路徑銀行；真實3181條，各E0合成798條，H歷史合成800條。少數原始生成路徑因無法在訓練分割內提供71日歷史，沿用既有排除紀錄。',
        '- Val：評分目標日期2018-01-01～2019-12-31，實際交易日2018-01-02～2019-12-31。24個完整21日窗口覆蓋503個交易日；最後一窗為覆蓋尾端而與前窗重疊1日。觀測與前置交易只使用各目標當時已知的過去資料，可來自2017年。',
        '- 合成Val：每條件合併2018原validation200條、2019原test200條，共400條；2019現在是validation，這輪沒有2019獨立test的宣稱。',
        '- 每個episode：50日觀測 → k日真實交易 →21日真實／合成交易；k每次reset均勻抽整數0–21。H/C/D一半來源episode為合成，前置日也是真實價格，所以實際合成交易步數約三分之一。',
        '- 16個環境、rollout每環境504步、minibatch504、每更新10 epochs；每個主要模型32更新＝258048步。reward=100×log扣費後淨值比，費率0.0025，初始全現金，初次建倉扣費。預測、來源旗標、k不進入觀測。',
        '- 同seed各組初始權重相同，C/D使用相同合成銀行、k序列及互動預算。不同seed的初始權重、路徑抽樣、k與D池RNG均不同。',
        '- 批次環境已逐步核對原版費用、持倉、觀測、終止與reset；D換池時讓未完成episode自然結束。續訓恢復策略、optimizer、環境與RNG的檢查另存。',
        '- 真實Val對每個目標平均全部22個k，再對原目標取平均作選模。合成Val與固定train樣本用事先固定、平衡的k。學習曲線的Train是固定64條真實／64條合成路徑的評估，並非整個訓練庫平均。','',
        '## D校準與模型選擇','',
        f'- 先用seed0及新Val比較重複1/2/4/8次，必要時測較大池。鎖定 pool={settings["environment_pool_size"]}、最短{settings["minimum_repeats"]}／最長{settings["maximum_repeats"]}次更新；改善門檻{settings["improvement_threshold"]:.6f}、退步門檻{settings["regression_threshold"]:.6f}。全部主實驗使用同一設定，不逐seed按事件績效調參。',
        '- 主結果使用2018–2019真實Val平均suffix reward最高的checkpoint，同分取較早，允許初始化被選中；final32為事先指定的敏感度分析。120個模型的兩套名單在事件策略損益計算前鎖定。',
        '- 三個E0條件仍使用各事件首21日實現報酬／年化波動率，不是整段事件條件。Test固定同一首21日的真實行情，由全現金開始、確定性動作、不插入額外交易日、不做分段預測或續訓。初次建倉成本正常計入。','',
        '| 情境 | 真實Test區間 | 條件市場報酬(未扣費) | 條件年化波動率 |',
        '|---|---|---:|---:|']
    for case,f in manifest['cases'].items():
        lines.append(f'| {LABELS[f["event"]]} | {f["condition_start"]}～{f["condition_end"]} | {100*f["actual_return"]:+.2f}% | {100*f["actual_volatility"]:.2f}% |')
    for selection,title in [('best_validation','主結果：驗證選模'),('final32','敏感度：最後32次更新')]:
        lines+=['',f'## {title}','','扣費後簡單報酬，平均±seed標準差；每組每情境n=10。全現金=0%。','',
            '| 情境 | R | H | C | D | 每日等權 |','|---|---:|---:|---:|---:|---:|']
        for case in m.CASES:
            f=manifest['cases'][case]
            d=summary[(summary.selection==selection)&(summary['case']==case)].set_index('group')
            values=[f'{d.loc[g,"mean_return_pct"]:+.2f}±{d.loc[g,"std_return_pct"]:.2f}%' for g in GROUPS]
            eq=baselines[(baselines['case']==case)&(baselines['group']=='equal_weight')].iloc[0]
            lines.append(f'| {LABELS[f["event"]]} | '+' | '.join(values)+f' | {100*eq.net_return:+.2f}% |')
        lines+=['','配對差異按同一seed相減；區間為95% t區間(df=9)，單位為百分點。','',
            '| 情境 | 比較 | 平均差 | 95%區間 | 勝／平／負 |','|---|---|---:|---|---:|']
        for case in m.CASES:
            f=manifest['cases'][case]
            for comparison in ['C-R','D-R','D-H','D-C']:
                a=pairs[(pairs.selection==selection)&(pairs['case']==case)&(pairs.comparison==comparison)].iloc[0]
                lines.append(f'| {LABELS[f["event"]]} | {comparison} | {a.mean_delta_pp:+.2f} | [{a.ci95_lower_pp:+.2f}, {a.ci95_upper_pp:+.2f}] | {int(a.wins)}／{int(a.ties)}／{int(a.losses)} |')
    lines+=['','## 主結果：風險、成本及初始化選中數','',
        '| 情境／組別 | 平均最大回撤 | 平均成本 | 平均股票曝險 | 選到初始化 |','|---|---:|---:|---:|---:|']
    for case in m.CASES:
        f=manifest['cases'][case]
        for group in GROUPS:
            a=summary[(summary.selection=='best_validation')&(summary['case']==case)&(summary['group']==group)].iloc[0]
            lines.append(f'| {LABELS[f["event"]]} / {group} | {a.mean_max_drawdown_pct:.2f}% | {a.mean_cost_pct:.3f}% | {a.mean_stock_exposure_pct:.2f}% | {int(a.initialization_selected_count)}/10 |')
    lines+=['','## 解讀限制與行為','',
        '- 95%區間只描述同一組生成資料、同一真實事件下PPO隨機種子的差異，不涵蓋生成器、預測或市場抽樣不確定性；10個seed不是10次獨立市場事件，三個情境也不合併成30個獨立市場樣本。',
        '- 條件由事後實現值建立，屬情境機制實驗；seed0的真實事件在先前開發已被查看，這輪不宣稱是從未看過事件的前瞻檢驗。這次沒有依事件結果調參、重抽條件或選checkpoint。',
        '- test_behavior.csv只用決策收盤日已知的市場訊號，統計其後交易的曝險調整，排除價格自然漂移。超過0.5個百分點才算有效。同一事件訊號在10個seed重複出現，不能視為10倍獨立行為樣本；固定曝險再平衡也可能產生同向交易，因此次數本身不能證明已學會指定交易行為。','',
        '## 檔案與驗證','',
        f'主實驗120模型、合計{120*32*m.BATCH:,}步。從tmux啟動至報告約{wall:.1f}分鐘，不含事前程式修改、測試與CPU計時。',
        '- test_summary.csv、paired_comparisons.csv：平均、標準差、配對95%區間與勝平負。',
        '- test_per_seed.csv、test_daily_trades.csv、test_behavior.csv：逐seed績效、交易、曝險與行為。',
        '- learning_curves.csv：全部主要模型的固定train/val評估；selected_checkpoints.csv、test_selection_locked.json：事先鎖定的模型及SHA-256。',
        '- main_job_summary.csv、checkpoint_inventory.json：訓練工作與checkpoint清單；D_parameters_locked.json及D_calibration_candidates.csv：seed0校準。',
        '- audit.json、integration_smoke/audit.json、environment_tests.log：資料與結果完整性、種子配對及續訓核對。',
        '- charts.pdf：學習曲線、真實Test淨值與曝險；陰影是seed標準差，並非市場信賴帶。','']
    lines += [f'- [{name}]({out/name})' for name in artifacts]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(out,workers):
    started=time.monotonic();assert (out/'training_complete.json').exists()
    manifest=lock_selection(out)
    (out/'event_test').mkdir(exist_ok=True)
    def one(seed):
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(out),'--seed',str(seed)]
        with (out/'event_test'/f'seed{seed:02d}.log').open('a') as handle:
            subprocess.run(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('Event test complete',seed,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,seed) for seed in m.SEEDS]
        for future in as_completed(futures):future.result()
    perf,days,signals,summary,pairs,baselines,learning=collect(out,manifest)
    artifacts=figures(out,days,summary,learning,manifest)
    report(out,summary,pairs,baselines,signals,manifest,artifacts)
    assert len(perf)==240 and len(learning)==3960
    assert np.isfinite(perf[['net_return','log_reward','max_drawdown','cost_pct_start_capital']]).all().all()
    assert not perf.duplicated(['case','group','seed','selection']).any()
    for case in m.CASES:
        for seed in m.SEEDS:
            c=json.loads((out/'jobs'/job_for(case,'C',seed)/'updates.json').read_text())
            d=json.loads((out/'jobs'/job_for(case,'D',seed)/'updates.json').read_text())
            for a,b in zip(c,d):
                for key in ['steps','completed_k_counts','prefix_steps','suffix_steps','completed_real_episodes','completed_synthetic_episodes']:assert a[key]==b[key],(case,seed,key)
    for name,info in manifest['models'].items():
        for selection in SELECTIONS:assert m.sha(info[selection]['path'])==info[selection]['sha256']
    bank=json.loads((out/'banks/manifest.json').read_text())
    for name,info in bank['banks'].items():assert m.sha(out/'banks'/f'{name}.npy')==info['sha256']
    assert json.loads((out/'integration_smoke/audit.json').read_text())['status']=='passed'
    plan=json.loads((out/'plan.json').read_text())
    assert plan['module_sha256']==m.sha(m.__file__)
    assert plan['random_k_dependency_sha256']==m.sha(m.k.__file__) and plan['original_helpers_sha256']==m.sha(m.k.original.__file__)
    result={'status':'passed','completed_utc':m.now(),'main_models':120,'main_seeds':m.SEEDS,
        'main_training_interactions':120*32*m.BATCH,'test_model_evaluations':240,
        'test_days_per_event':21,'validation_dates':['2018-01-01','2019-12-31'],
        'D_locked_before_main':True,'same_seed_initialization_and_C_D_k_streams_match':True,
        'different_seeds_have_distinct_initializations':True,'event_selection_manifest_frozen_before_test':True,
        'no_2019_independent_test_claim':True,'module_sha256':m.sha(m.__file__),'finalizer_sha256':m.sha(__file__),
        'postprocessing_seconds':time.monotonic()-started,
        'artifacts_sha256':{name:m.sha(out/name) for name in artifacts+['REPORT.md','charts.pdf','test_summary.csv',
            'paired_comparisons.csv','test_per_seed.csv','test_daily_trades.csv','test_behavior.csv','learning_curves.csv','selected_checkpoints.csv']}}
    m.write_json(out/'audit.json',result);m.write_json(out/'status.json',{**result,'status':'complete'})
    (out/'finalizer_source_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.seed is not None:test_worker(args.out,args.seed)
    else:main(args.out,args.workers)
