"""Verify steps 1-3 outputs and append the measured C/D calibration outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main(root):
    root=root.resolve()
    state=json.loads((root/'status.json').read_text());assert state['status']=='complete'
    plan=json.loads((root/'plan.json').read_text())
    bank=json.loads((root/'banks/manifest.json').read_text())
    assert bank['observation_keys']==['state','last_action'] and bank['k_max']==0
    assert bank['commission']==.0025
    integration=json.loads((root/'integration_smoke/audit.json').read_text())
    assert integration['status']=='passed' and integration['optimizer_state_exactly_identical']
    assert json.loads((root/'environment_tests.json').read_text())['passed']==10
    for name,record in bank['banks'].items():
        assert sha(root/'banks'/f'{name}.npy')==record['sha256'],name
        values=np.load(root/'banks'/f'{name}.npy',mmap_mode='r')
        assert values.shape==(record['n'],71,10,3)
        assert np.isfinite(values).all() and (values>0).all()
        assert (values[...,2]<=values[...,0]).all() and (values[...,0]<=values[...,1]).all()
    primary={};identities=set();rows=[];total_steps=0
    for p in sorted((root/'jobs').glob('*/status.json')):
        status=json.loads(p.read_text());config=status['config']
        assert status['status']=='complete'
        assert config['source_code_sha256']==plan['code_sha256']
        assert config['commission']==.0025 and config['k_max']==0 and config['seed']==0
        assert config['banks']==str(root/'banks')
        logs=pd.read_csv(p.parent/'updates.csv')
        count=config['updates']
        assert len(logs)==count
        np.testing.assert_array_equal(logs.steps.to_numpy(),np.arange(1,count+1)*8064)
        np.testing.assert_array_equal(logs.completed_real_episodes+logs.completed_synthetic_episodes,logs.steps//21)
        if config['group']=='R':assert (logs.completed_synthetic_episodes==0).all()
        else:np.testing.assert_array_equal(logs.completed_real_episodes,logs.completed_synthetic_episodes)
        with zipfile.ZipFile(p.parent/'latest_model.zip') as archive:
            data=json.loads(archive.read('data'))
            assert data['num_timesteps']==count*8064
            assert 'policy.optimizer.pth' in archive.namelist()
        resume=json.loads((p.parent/'resume.json').read_text())
        assert resume['update']==count
        assert all(s['day']==0 for s in resume['envs'])
        identities.add(status['initialization']['policy_sha256'])
        actual=json.loads((p.parent/'validation_returns.json').read_text())
        assert len(actual)==count+1 and actual[0]['update']==0
        assert all(len(v['real'])==bank['banks']['real_validation']['n'] for v in actual)
        assert all(len(v['synthetic'])==(0 if config['group']=='R' else 200) for v in actual)
        best_index=int(np.argmax([np.mean(v['real']) for v in actual]))
        with zipfile.ZipFile(p.parent/'best_model.zip') as archive:
            best_metadata=json.loads(archive.read('data'))
            assert best_metadata['num_timesteps']==actual[best_index]['update']*8064
        np.testing.assert_allclose(status['best_validation_score'],np.mean(actual[best_index]['real']),rtol=1e-12,atol=1e-12)
        total_steps+=status['steps']
        if config['updates']==32:primary[p.parent.name]=(config,logs,actual)
        rows.append({'name':p.parent.name,'steps':int(status['steps']),'updates':count,
                     'checkpoint_sha256':sha(p.parent/'latest_model.zip'),'initial_policy_sha256':status['initialization']['policy_sha256']})
    assert len(identities)==1
    selected=json.loads((root/'D_selected_parameters.json').read_text())
    assert selected['minimum_repeats']<=selected['maximum_repeats']
    assert selected['environment_pool_size'] in (16,32,64)
    evidence=json.loads((root/'D_calibration_evidence.json').read_text())
    errors=[]
    for p in (root/'jobs').glob('*_D_repeat*_pool16/updates.json'):
        errors.extend(r['validation_gain_se'] for r in json.loads(p.read_text()))
    np.testing.assert_allclose(selected['improvement_threshold'],max(float(np.median(errors)),1e-10))
    np.testing.assert_allclose(selected['regression_threshold'],max(float(np.quantile(errors,.9)),1e-10))
    for row in evidence:
        p=root/'jobs'/f"{row['case']}_D_repeat{row['repeat']}_pool{row['pool_size']}"/'validation_returns.json'
        c=root/'jobs'/f"{row['case']}_C"/'validation_returns.json'
        def score(file):return np.mean([v['real'] for v in json.loads(file.read_text()) if 5<=v['update']<=8],axis=0)
        difference=score(p)-score(c)
        np.testing.assert_allclose(difference,row['episode_differences'],rtol=1e-12,atol=1e-12)
    repeat_vectors={k:np.mean([r['episode_differences'] for r in evidence if r['repeat']==k and r['pool_size']==16],axis=0) for k in (1,2,4,8)}
    stderr=lambda v:float(np.std(v,ddof=1)/np.sqrt(len(v)))
    best_repeat=max(repeat_vectors,key=lambda k:float(np.mean(repeat_vectors[k])))
    assert selected['best_fixed_repeat']==best_repeat==selected['maximum_repeats']
    smallest=min(k for k,v in repeat_vectors.items() if k<=best_repeat and np.mean(repeat_vectors[best_repeat]-v)<=stderr(repeat_vectors[best_repeat]-v)+1e-12)
    assert selected['minimum_repeats']==smallest
    pool_vectors={}
    for pool in (16,32,64):
        data=[r['episode_differences'] for r in evidence if r['repeat']==best_repeat and r['pool_size']==pool]
        if data:pool_vectors[pool]=np.mean(data,axis=0)
    best_pool=max(pool_vectors,key=lambda k:float(np.mean(pool_vectors[k])))
    expected_pool=min(k for k,v in pool_vectors.items() if np.mean(pool_vectors[best_pool]-v)<=stderr(pool_vectors[best_pool]-v)+1e-12)
    assert selected['environment_pool_size']==expected_pool
    assert len(rows)==(54 if selected['larger_pool_triggered'] else 42)
    outcomes=[];best_outcomes=[]
    cases=[f'{event}_{error}_f0' for event in ['crash_2020','rebound_2020','bear_2022'] for error in ('E0','E1')]
    for case in cases:
        _,clog,cret=primary[case+'_C'];_,dlog,dret=primary[case+'_D_adaptive']
        # Equal-budget final policies, no best-checkpoint hindsight selection.
        delta=np.asarray(dret[-1]['real'])-np.asarray(cret[-1]['real'])
        c_window=np.mean([v['real'] for v in cret if 29<=v['update']<=32],axis=0)
        d_window=np.mean([v['real'] for v in dret if 29<=v['update']<=32],axis=0)
        outcomes.append({'case':case,'C_final_real2018_logreward':float(np.mean(cret[-1]['real'])),
                         'D_final_real2018_logreward':float(np.mean(dret[-1]['real'])),
                         'D_minus_C_final_mean':float(delta.mean()),'D_minus_C_mean_over_updates29_32':float((d_window-c_window).mean()),
                         'n_real_validation_episodes':len(delta),'switch_reasons':dlog.switch_reason.value_counts().to_dict()})
        ci=int(np.argmax([np.mean(v['real']) for v in cret]));di=int(np.argmax([np.mean(v['real']) for v in dret]))
        cb=float(np.mean(cret[ci]['real']));db=float(np.mean(dret[di]['real']))
        best_outcomes.append({'case':case,'C_selected_update':cret[ci]['update'],'D_selected_update':dret[di]['update'],
                              'C_best_real2018_logreward':cb,'D_best_real2018_logreward':db,'D_minus_C_best':db-cb})
    pd.DataFrame(outcomes).to_csv(root/'adaptive_D_vs_C.csv',index=False)
    pd.DataFrame(best_outcomes).to_csv(root/'best_checkpoint_D_vs_C.csv',index=False)
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for ax,case in zip(axes.flat,cases):
        event=case.rsplit('_',2)[0]
        for group,name,color in [('R',event+'_R','#777777'),('H',event+'_H','#E69F00'),('C',case+'_C','#0072B2'),('D',case+'_D_adaptive','#009E73')]:
            _,logs,returns=primary[name]
            ax.plot(logs.steps,logs.real_validation_score,label=group,color=color,lw=1.3)
            selected_index=int(np.argmax([np.mean(v['real']) for v in returns]))
            ax.scatter([returns[selected_index]['update']*8064],[np.mean(returns[selected_index]['real'])],color=color,s=24,zorder=4)
        ax.set(title=case,xlabel='環境互動數',ylabel='2018 真實驗證平均 21 日 log reward')
        ax.legend(ncols=4,frameon=False)
        ax.grid(alpha=.12)
    fig.suptitle('seed=0：2018 驗證曲線；圓點為選模 checkpoint；沒有真實事件績效',fontsize=14)
    fig.savefig(root/'real2018_learning_curves.png',dpi=170);plt.close(fig)
    report=root/'REPORT.md';text=report.read_text()
    marker='## 獨立核對與 adaptive D 結果'
    if marker in text:text=text.split(marker)[0]
    extra=[marker,'',f'10 項環境／記帳測試及實際 checkpoint 續訓一致性通過。共核對 {len(rows)} 個已完成工作、{total_steps:,} 次訓練互動；所有工作初始策略參數雜湊一致。',
           '資料檔案、每更新步數、完整 episode、來源 50% 比例、checkpoint 內 optimizer、RNG 與環境存檔均核對。',
           f'真實驗證有 {bank["banks"]["real_validation"]["n"]} 條目標互不重疊的 2018 episode，評分目標從 {bank["banks"]["real_validation"]["first_target"]} 至 {bank["banks"]["real_validation"]["last_target"]}。為使歷史與未來完整位於 2018，前 50 日僅作觀測；尾端不足 21 日不組成 episode。這不是全年報酬；仍是小樣本且所有調參共用同一驗證年度，不能當成獨立泛化檢定。','',
           '每個完整更新包含每個活躍環境 24 條完整 21 日 episode。16 路徑的固定 D 池中，每條路徑在一次更新內被走 24 次；重複 k 個更新週期代表同一路徑最多 24×k 次 episode 訪問，且每次更新都重新用新策略收集 rollout。這個計數必須與「k 個 PPO 更新」分開解讀。','',
           '### 排程選擇的彙總證據','',
           '| 固定池16：重複更新數 | 相對 C 的配對分數改善 |','|---:|---:|']
    for k,v in repeat_vectors.items():extra.append(f'| {k} | {float(np.mean(v)):+.6f} |')
    extra+=['',f'固定較佳重複數 {best_repeat}，再比較池大小：','',
            '| 路徑池大小 | 相對 C 的配對分數改善 |','|---:|---:|']
    for k,v in pool_vectors.items():extra.append(f'| {k} | {float(np.mean(v)):+.6f} |')
    extra+=['','以上分數先對同一條真實 2018 驗證 episode 的六個條件取平均，再取驗證 episode 平均；不把六種條件重複視為六倍獨立市場樣本。池大小是固定最佳重複數後的分段搜索，並非完整池大小×重複次數的全因子搜索。','']
    if selected['minimum_repeats']==selected['maximum_repeats']:
        k=selected['minimum_repeats']
        extra += [f'**本輪選出的範圍為 [{k}, {k}]，實際排程是固定每 {k} 個更新換池。** 平台與驗證退步門檻仍作記錄，但在此退化範圍下不會改變換池時點；本輪沒有證明自適應提前停用比固定週期有額外效益。','']
    extra += [
           '| 條件 | C 最終驗證分數 | D 最終驗證分數 | D−C 最終差 | 更新29–32平均差 |','|---|---:|---:|---:|---:|']
    for row in outcomes:
        extra.append(f"| {row['case']} | {row['C_final_real2018_logreward']:.6f} | {row['D_final_real2018_logreward']:.6f} | {row['D_minus_C_final_mean']:+.6f} | {row['D_minus_C_mean_over_updates29_32']:+.6f} |")
    better=sum(r['D_minus_C_final_mean']>0 for r in outcomes)
    extra+=['',f'在固定 258,048 次互動的最後策略比較，D 的真實 2018 驗證分數高於 C 的條件為 {better}/6；其餘未改善結果完整保留，不把選出參數當成 D 已普遍有效。',
            '表中分數單位為 21 日累積的 100×log 扣費後資產比，不是事件報酬率，也不是年化報酬率。','',
            '### 依既定 2018 選模規則保存的最佳 checkpoint','',
            '| 條件 | C 選定更新 | D 選定更新 | C 最佳驗證 | D 最佳驗證 | D−C |','|---|---:|---:|---:|---:|---:|']
    for row in best_outcomes:
        extra.append(f"| {row['case']} | {row['C_selected_update']} | {row['D_selected_update']} | {row['C_best_real2018_logreward']:.6f} | {row['D_best_real2018_logreward']:.6f} | {row['D_minus_C_best']:+.6f} |")
    best_better=sum(r['D_minus_C_best']>0 for r in best_outcomes)
    extra+=['',f'按各自最佳 2018 checkpoint 比較，D 較 C 高的條件也是 {best_better}/6。這是同一驗證資料上的選模分數，不能當作未見測試績效；選模資料未使用真實事件或 2019。',
            '部分曲線長訓練後回落，最佳與最後 checkpoint 都保留；此結果提示在下一階段確認訓練長度與收斂，不能以初期校準改善直接外推至長期訓練。','',
            f'![2018驗證曲線]({root/"real2018_learning_curves.png"})','']
    report.write_text(text+'\n'+'\n'.join(extra))
    audit={'status':'passed','jobs':len(rows),'total_training_steps':total_steps,'initial_policy_sha256':next(iter(identities)),
           'scope':'steps1-3 only; no 2019 policy evaluation or real event backtests','adaptive_outcomes':outcomes,'best_checkpoint_outcomes':best_outcomes,'checkpoints':rows}
    (root/'independent_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    shutil.copy2(Path(__file__),root/'source_snapshot/scripts/audit_timediff_pilot.py')
    snapshots=root/'source_snapshot'
    (snapshots/'sha256.json').write_text(json.dumps({str(p.relative_to(snapshots)):sha(p) for p in snapshots.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
    # The finalizer and log streams may append after this audit returns.
    inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('inventory.sha256.json','finalization.json') and p.suffix!='.log'}
    (root/'inventory.sha256.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'status':'passed','jobs':len(rows),'training_steps':total_steps,'D_better_than_C_cases':better,'report':str(report)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
