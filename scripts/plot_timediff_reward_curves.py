"""Evaluate frozen seed-0 PPO checkpoints and plot auditable reward curves.

Writes a new diagnostic directory only; never trains or selects a checkpoint.
Real test prices are restricted to 2019. Actual 2020/2022 events are not scored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from finrl.experiments import timediff_scenario_pilot as pilot

STUDY = Path('/mnt/sda/cwei1213/timediff_finrl/steps1_3_20261115')
OUT = STUDY / 'reward_curves_20261115'
GROUPS = ['R', 'H', 'C', 'D']
EVENT_LABELS = {'crash_2020': '2020 crash', 'rebound_2020': '2020 rebound', 'bear_2022': '2022 bear'}
SPLITS = ['train', 'validation', 'test']


def job_for(case, group):
    event = case.rsplit('_', 2)[0]
    return f'{event}_{group}' if group in ('R', 'H') else f'{case}_{group}' + ('_adaptive' if group == 'D' else '')


def case_label(case):
    event, error, _ = case.rsplit('_', 2)
    return f'{EVENT_LABELS[event]} | {error}'


def paths_to_prices(path):
    with np.load(path, allow_pickle=False) as z:
        assert z['tickers'].tolist() == pilot.TICKERS
        index = z['history_index']
        history = np.stack([z[f'history_{k}'][index, -50:] for k in ('close', 'high', 'low')], axis=-1)
        future = np.stack([z[k] for k in ('close', 'high', 'low')], axis=-1)
        np.testing.assert_array_equal(history[:, -1, :, 0], z['anchor_close'])
        assert all(str(d).startswith('2019-') for d in z['anchor_dates'])
        values = np.concatenate([history, future], axis=1)
        rounding = pilot.price_roundoff(values)
        metadata = {'n': len(values), 'source': str(path), 'source_sha256': pilot.sha(path),
                    'candidate_ids': z['candidate_id'].tolist(), 'background_ids': z['background_id'].tolist(),
                    'anchor_dates': z['anchor_dates'].tolist(), **rounding}
    return values, metadata


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'evaluation_manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    original = json.loads((STUDY / 'banks/manifest.json').read_text())
    source = Path(original['source'])
    generation = json.loads((source / 'manifest.json').read_text())
    assert pilot.sha(generation['data']) == original['data_sha256'] == generation['data_sha256']
    assert generation['generator_metadata']['args']['lambda_condition'] == .5
    banks = out / 'evaluation_banks'
    banks.mkdir(exist_ok=True)
    manifest = {'created_utc': pilot.now(), 'status': 'frozen_before_checkpoint_evaluation',
        'study': str(STUDY), 'generator_lambda': .5, 'rl_seed': 0, 'forecast_seed': 0,
        'script_sha256': pilot.sha(__file__), 'original_bank_manifest_sha256': pilot.sha(STUDY / 'banks/manifest.json'),
        'generation_manifest_sha256': pilot.sha(source / 'manifest.json'),
        'price_source_sha256': generation['data_sha256'],
        'reward': 'mean episode sum of daily rewards = mean 100*ln(net V_21/V_0); not simple return percent',
        'fees': .0025, 'episode_days': 21, 'history_days': 50,
        'deterministic_policy': True, 'initial_holdings': 'all cash, initial purchase charged',
        'no_training_or_checkpoint_selection': True, 'actual_event_backtests': False,
        'train_selection': '64 equally spaced real bank indices; 64 fixed RNG synthetic indices; fixed before evaluation',
        'train_synthetic_selection_seed': 20261115,
        'validation': 'reuse original deterministic 2018 real 9 + synthetic 200 scores; verify initial and final policies',
        'test': 'new diagnostic evaluation: real 2019 9 episodes + retained synthetic test 200 per condition; no tuning',
        'matched_mixture': 'R: real only; H/C/D: 0.5*mean(real)+0.5*mean(group-matched synthetic)',
        'common_real': 'all groups evaluated on the identical fixed real train/validation/test episodes',
        'rollout': 'SB3 rollout/ep_rew_mean: stochastic policy, latest 100 complete training episodes, logged before current update',
        'banks': {}, 'source_files': {}, 'job_mapping': {}}
    for name, meta in original['banks'].items():
        bank = STUDY / 'banks' / f'{name}.npy'
        assert pilot.sha(bank) == meta['sha256']
        manifest['source_files'][str(bank)] = meta['sha256']
    # Filter before pivoting or constructing windows: no actual event target enters evaluation.
    frame = pd.read_csv(generation['data'], parse_dates=['date'])
    frame = frame[(frame.date >= '2005-01-01') & (frame.date <= '2019-12-31')]
    wide = {k: frame.pivot(index='date', columns='tic', values=k)[pilot.TICKERS] for k in ('close', 'high', 'low')}
    dates = wide['close'].index
    prices = np.stack([wide[k].to_numpy() for k in ('close', 'high', 'low')], axis=-1)
    train_starts = np.arange(np.flatnonzero(dates >= pd.Timestamp('2005-01-01'))[0] + 50,
                             np.flatnonzero(dates <= pd.Timestamp('2017-12-31'))[-1] - 20 + 1)
    real_train = np.load(STUDY / 'banks/real_train.npy', mmap_mode='r')
    indices = np.linspace(0, len(real_train)-1, 64, dtype=int)
    assert len(train_starts) == len(real_train) and np.diff(indices).min() >= 21
    selected = np.array(real_train[indices])
    # Confirm reconstructed target dates refer to the exact saved training episodes.
    for i, start in zip(indices, train_starts[indices]):
        reference = prices[start-50:start+21].copy()
        pilot.price_roundoff(reference)
        np.testing.assert_array_equal(reference, real_train[i])
    np.save(banks / 'real_train.npy', selected)
    manifest['banks']['real_train'] = {'n': len(indices), 'selected_bank_indices': indices.tolist(),
        'targets': [[str(dates[s].date()), str(dates[s+20].date())] for s in train_starts[indices]],
        'sha256': pilot.sha(banks / 'real_train.npy')}
    eligible = np.flatnonzero((dates >= pd.Timestamp('2019-01-01')) & (dates <= pd.Timestamp('2019-12-31')))
    starts = np.arange(eligible[0]+50, eligible[-1]-20+1, 21)
    values = np.stack([prices[s-50:s+21] for s in starts])
    rounding = pilot.price_roundoff(values)
    assert len(values) == 9
    np.save(banks / 'real_test.npy', values)
    manifest['banks']['real_test'] = {'n': len(values), 'history_first_date': str(dates[starts[0]-50].date()),
        'targets': [[str(dates[s].date()), str(dates[s+20].date())] for s in starts],
        'target_stride': 21, 'sha256': pilot.sha(banks / 'real_test.npy'), **rounding}
    for fid in ['H_historical'] + pilot.CASES:
        synthetic = np.load(STUDY / 'banks' / f'{fid}_train.npy', mmap_mode='r')
        # Same predeclared indices for C/D; path choice never depends on rewards.
        index = np.sort(np.random.default_rng(20261115).choice(len(synthetic), 64, replace=False))
        np.save(banks / f'{fid}_train.npy', synthetic[index])
        manifest['banks'][f'{fid}_train'] = {'n': len(index), 'selected_bank_indices': index.tolist(),
            'sha256': pilot.sha(banks / f'{fid}_train.npy')}
        path = source / fid / 'test/paths.npz'
        summary = json.loads((path.parent / 'summary.json').read_text())
        assert summary['status'] == 'complete' and pilot.sha(path) == summary['artifacts_sha256']['paths.npz']
        values, metadata = paths_to_prices(path)
        assert len(values) == 200
        np.save(banks / f'{fid}_test.npy', values)
        metadata['sha256'] = pilot.sha(banks / f'{fid}_test.npy')
        manifest['banks'][f'{fid}_test'] = metadata
        manifest['source_files'][str(path)] = pilot.sha(path)
    jobs = []
    for case in pilot.CASES:
        manifest['job_mapping'][case] = {g: job_for(case, g) for g in GROUPS}
        for group in GROUPS:
            name = job_for(case, group)
            if name in jobs:
                continue
            jobs.append(name)
            directory = STUDY / 'jobs' / name
            config = json.loads((directory / 'job.json').read_text())
            assert config['updates'] == 32 and config['group'] == group and config['seed'] == 0
            assert json.loads((directory / 'status.json').read_text())['status'] == 'complete'
            for file in ['job.json', 'status.json', 'updates.csv', 'validation_returns.json', 'ppo/progress.csv',
                         'initial_model.zip'] + [f'checkpoint_{u:03d}.zip' for u in range(1, 33)]:
                manifest['source_files'][str(directory / file)] = pilot.sha(directory / file)
    manifest['jobs'] = jobs
    assert len(jobs) == 18
    pilot.write_json(manifest_path, manifest)
    (out / 'source_snapshot.py').write_text(Path(__file__).read_text())
    return manifest


def save_npz(path, payload):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    temp.replace(path)


def worker(out, name):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = json.loads((out / 'evaluation_manifest.json').read_text())
    job = STUDY / 'jobs' / name
    config = json.loads((job / 'job.json').read_text())
    val = json.loads((job / 'validation_returns.json').read_text())
    assert [v['update'] for v in val] == list(range(33))
    folder = out / 'evaluations'
    folder.mkdir(exist_ok=True)
    target = folder / f'{name}.npz'
    status = folder / f'{name}.json'
    banks = out / 'evaluation_banks'
    components = {'real_train': np.load(banks / 'real_train.npy'), 'real_test': np.load(banks / 'real_test.npy')}
    fid = 'H_historical' if config['group'] == 'H' else config['case']
    if config['group'] != 'R':
        components.update({f'synthetic_{split}': np.load(banks / f'{fid}_{split}.npy') for split in ('train', 'test')})
    slices = {}
    offset = 0
    for key, value in components.items():
        slices[key] = slice(offset, offset + len(value))
        offset += len(value)
    combined = np.concatenate(list(components.values()))
    data = {key: [] for key in components}
    data.update(real_validation=[], synthetic_validation=[], updates=[], checkpoint_sha256=[])
    prior = None
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            data = {key: z[key].tolist() for key in data}
        prior = json.loads(status.read_text())
        assert prior['manifest_sha256'] == pilot.sha(out / 'evaluation_manifest.json')
    initial = len(data['updates'])
    started = time.monotonic()
    checks = prior.get('validation_reproduction_checks_this_run', []) if prior else []
    for update in range(initial, 33):
        checkpoint = job / ('initial_model.zip' if update == 0 else f'checkpoint_{update:03d}.zip')
        assert pilot.sha(checkpoint) == manifest['source_files'][str(checkpoint)]
        model = PPO.load(checkpoint, device='cpu')
        assert model.num_timesteps == update * pilot.BATCH
        values = pilot.evaluate(model, combined)
        assert np.isfinite(values).all()
        for key, part in slices.items():
            data[key].append(values[part].tolist())
        data['real_validation'].append(val[update]['real'])
        data['synthetic_validation'].append(val[update]['synthetic'])
        data['updates'].append(update)
        data['checkpoint_sha256'].append(pilot.sha(checkpoint))
        if update in (0, 32):
            actual = pilot.evaluate(model, np.load(STUDY / 'banks/real_validation.npy'))
            np.testing.assert_allclose(actual, val[update]['real'], atol=1e-5, rtol=1e-6)
            if config['group'] != 'R':
                actual = pilot.evaluate(model, np.load(STUDY / 'banks' / f'{fid}_validation.npy'))
                np.testing.assert_allclose(actual, val[update]['synthetic'], atol=1e-5, rtol=1e-6)
            checks.append(update)
        save_npz(target, data)
        pilot.write_json(status, {'status': 'complete' if update == 32 else 'running', 'job': name,
            'manifest_sha256': pilot.sha(out / 'evaluation_manifest.json'), 'completed_checkpoints': update+1,
            'validation_reproduction_checks_this_run': checks, 'elapsed_seconds_this_run': time.monotonic()-started,
            'updated_utc': pilot.now()})
        if update % 8 == 0:
            print(f'{name}: checkpoint {update}/32 evaluated', flush=True)
        del model


def tables(out):
    manifest = json.loads((out / 'evaluation_manifest.json').read_text())
    rows, raw, trends = [], [], []
    for case in pilot.CASES:
        for group in GROUPS:
            name = job_for(case, group)
            directory = STUDY / 'jobs' / name
            progress = pd.read_csv(directory / 'ppo/progress.csv')
            updates = pd.read_csv(directory / 'updates.csv')
            assert len(progress) == len(updates) == 32
            for i, (_, row) in enumerate(progress.iterrows(), 1):
                assert row['time/total_timesteps'] == i * pilot.BATCH
                raw.append({'case': case, 'group': group, 'job': name, 'collection_update': i,
                    'logged_collection_steps': i*pilot.BATCH, 'policy_checkpoint_used': i-1,
                    'policy_trained_steps': (i-1)*pilot.BATCH, 'reward': row['rollout/ep_rew_mean'],
                    'definition': 'latest100_stochastic_training_episode_rewards',
                    'replay_gain': updates.iloc[i-1]['replay_gain'] if group == 'D' else None,
                    'switch_reason': updates.iloc[i-1]['switch_reason']})
            if group == 'D':
                v = progress['rollout/ep_rew_mean']
                trends.append({'case': case, 'first_rollout': v.iloc[0], 'last_rollout': v.iloc[-1],
                    'first4_mean': v.head(4).mean(), 'last4_mean': v.tail(4).mean(),
                    'last4_minus_first4': v.tail(4).mean()-v.head(4).mean(),
                    'positive_same_pool_update_gains': int((updates.replay_gain > 0).sum()),
                    'updates': len(updates), 'mean_same_pool_update_gain': updates.replay_gain.mean()})
            with np.load(out / 'evaluations' / f'{name}.npz', allow_pickle=False) as z:
                assert z['updates'].tolist() == list(range(33))
                for split in SPLITS:
                    real = z[f'real_{split}']
                    assert real.shape == (33, 64 if split == 'train' else 9)
                    synthetic = z[f'synthetic_{split}'] if group != 'R' else None
                    if synthetic is not None:
                        assert synthetic.shape == (33, 64 if split == 'train' else 200)
                    for i in range(33):
                        real_mean = float(real[i].mean())
                        syn_mean = float(synthetic[i].mean()) if synthetic is not None else None
                        mixture = real_mean if syn_mean is None else .5*real_mean+.5*syn_mean
                        rows.append({'case': case, 'group': group, 'job': name, 'update': i,
                            'steps': i*pilot.BATCH, 'split': split, 'real_n': real.shape[1],
                            'synthetic_n': 0 if synthetic is None else synthetic.shape[1],
                            'real_reward': real_mean, 'synthetic_reward': syn_mean, 'matched_reward': mixture,
                            'checkpoint_sha256': str(z['checkpoint_sha256'][i])})
    curves = pd.DataFrame(rows)
    raw = pd.DataFrame(raw)
    trends = pd.DataFrame(trends)
    curves.to_csv(out / 'checkpoint_reward_curves.csv', index=False)
    raw.to_csv(out / 'rollout_training_rewards.csv', index=False)
    trends.to_csv(out / 'D_training_trends.csv', index=False)
    final = curves[curves['update'] == 32].merge(
        curves[curves['update'] == 0][['case', 'group', 'split', 'real_reward', 'matched_reward']],
        on=['case', 'group', 'split'], suffixes=('_final', '_initial'))
    final['real_improvement'] = final.real_reward_final-final.real_reward_initial
    final['matched_improvement'] = final.matched_reward_final-final.matched_reward_initial
    final.to_csv(out / 'initial_vs_final.csv', index=False)
    paired = curves[curves['group'].isin(['C', 'D'])].pivot(index=['case','update','steps','split'],
        columns='group', values=['real_reward','matched_reward']).reset_index()
    paired.columns = ['_'.join(filter(None, x)) if isinstance(x, tuple) else x for x in paired.columns]
    for kind in ('real', 'matched'):
        paired[f'{kind}_D_minus_C'] = paired[f'{kind}_reward_D']-paired[f'{kind}_reward_C']
    paired.to_csv(out / 'paired_D_vs_C.csv', index=False)
    return curves, raw, trends, final, paired


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.dpi': 170})
    (out / 'reporting_source_snapshot.py').write_text(Path(__file__).read_text())
    curves, raw, trends, final, paired = tables(out)
    colors = {'train': '#0072B2', 'validation': '#D55E00', 'test': '#009E73'}
    group_colors = {'R': '#666666', 'H': '#0072B2', 'C': '#E69F00', 'D': '#CC79A7'}
    dashes = {'R':'-', 'H':'--', 'C':'-.', 'D':':'}
    def style(ax):
        ax.axhline(0, color='#aaaaaa', linewidth=.7)
        ax.grid(axis='y', alpha=.2)
        ax.set_xlim(0, 32*pilot.BATCH)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: f'{x/1000:.0f}k'))
        ax.set_xlabel('PPO training interactions')
        ax.set_ylabel('Mean 21-day episode reward')
    artifacts = []
    with PdfPages(out / 'all_reward_curves.pdf') as pdf:
        def save(fig, name):
            fig.savefig(out / name, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            artifacts.append(name)
            plt.close(fig)
        for measure, prefix, description in [
            ('matched_reward', '', 'Group-matched data: R real only; H/C/D 50% real + 50% synthetic'),
            ('real_reward', 'common_real_', 'Identical real data for all groups; synthetic episodes excluded')]:
            for group in GROUPS:
                selected_cases = [f'{event}_E0_f0' for event in pilot.EVENTS] if group in ('R','H') else pilot.CASES
                nrows, ncols = (1,3) if len(selected_cases) == 3 else (3,2)
                fig, axes = plt.subplots(nrows, ncols, figsize=(16,6) if nrows==1 else (13.5,11), squeeze=False)
                for ax, case in zip(axes.flat, selected_cases):
                    for split in SPLITS:
                        data = curves[(curves['case']==case)&(curves['group']==group)&(curves['split']==split)]
                        label = {'train':'Train (fixed 64/source)', 'validation':'Val (2018)', 'test':'Test (2019)'}[split]
                        ax.plot(data.steps, data[measure], color=colors[split], marker='o', markersize=2.6,
                                linewidth=1.6, label=label)
                    ax.set_title(case_label(case) if group in ('C','D') else EVENT_LABELS[case.rsplit('_',2)[0]]+' | E0/E1 shared')
                    style(ax)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.895 if nrows==1 else .94), ncol=3, frameon=False)
                fig.suptitle(f'{group}: deterministic train / validation / test reward', fontsize=17, y=1.01)
                fig.text(.5,.94 if nrows==1 else .965,description,ha='center',fontsize=10)
                fig.text(.5,.008,'Reward = 100 ln(net V21/V0), fee 0.0025 incl. first purchase. Seed 0; test is diagnostic; no test-based selection.',
                         ha='center', fontsize=9)
                fig.tight_layout(rect=(0,.035,1,.84 if nrows==1 else .9))
                save(fig, f'{prefix}{group}_train_val_test.png')
        fig, axes = plt.subplots(3,2,figsize=(13.5,11))
        for ax, case in zip(axes.flat,pilot.CASES):
            for group in GROUPS:
                data = raw[(raw['case']==case)&(raw['group']==group)]
                ax.plot(data.logged_collection_steps, data.reward, label=group, color=group_colors[group],
                        linestyle=dashes[group], linewidth=1.6, marker='o', markersize=2.5)
            style(ax)
            ax.set_title(case_label(case))
            ax.set_xlabel('Collected interactions (rollout before current PPO update)')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.96),frameon=False)
        fig.suptitle('Actual stochastic training reward | R / H / C / D',fontsize=17,y=1.01)
        fig.text(.5,.012,'SB3 latest 100 completed episodes; raw values, no smoothing. H/C/D use changing 50/50 real/synthetic episode streams.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.92))
        save(fig,'actual_training_reward_R_H_C_D.png')
        fig,axes=plt.subplots(3,2,figsize=(13.5,10))
        for ax,case in zip(axes.flat,pilot.CASES):
            data=raw[(raw['case']==case)&(raw['group']=='D')]
            ax.bar(data.collection_update,data.replay_gain,color=['#009E73' if v>0 else '#D55E00' for v in data.replay_gain],width=.8)
            ax.axhline(0,color='#666666',linewidth=.8)
            ax.set_title(case_label(case))
            ax.set_xlabel('Completed PPO update')
            ax.set_ylabel('Same-pool reward gain: after - before')
            ax.grid(axis='y',alpha=.2)
            row=trends[trends['case']==case].iloc[0]
            ax.text(.02,.96,f'Positive: {int(row.positive_same_pool_update_gains)}/32 | mean {row.mean_same_pool_update_gain:+.3f}',
                    transform=ax.transAxes,va='top',fontsize=10)
        fig.suptitle('D: did each PPO update improve its retained training paths?',fontsize=16,y=1.01)
        fig.text(.5,.01,'Deterministic, matched paths within each update. Pool changes every 2 updates; gains cannot be summed across pools.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.98))
        save(fig,'D_same_pool_update_gains.png')
    write_report(out, trends, final, paired, artifacts)
    return artifacts


def write_report(out, trends, final, paired, artifacts):
    manifest = json.loads((out / 'evaluation_manifest.json').read_text())
    lines=['# TimeDiff λ=0.5：PPO reward curves（seed=0）', '',
        'D 的實際訓練 reward 在六個條件均上升：最後 4 次 rollout 的平均皆高於最初 4 次。此判斷是描述性比較，不能取代獨立驗證或證明跨種子改善。', '',
        '| 情境 | 前4次平均 | 後4次平均 | 改善 | 同一訓練池更新後改善次數 |',
        '|---|---:|---:|---:|---:|']
    for case in pilot.CASES:
        r=trends[trends['case']==case].iloc[0]
        lines.append(f'| {case_label(case).replace("|", "／")} | {r.first4_mean:.3f} | {r.last4_mean:.3f} | {r.last4_minus_first4:+.3f} | {int(r.positive_same_pool_update_gains)}/32 |')
    d_final = final[final['group']=='D']
    val_positive = int((d_final[d_final['split']=='validation'].real_improvement>0).sum())
    test_positive = int((d_final[d_final['split']=='test'].real_improvement>0).sum())
    lines += ['', f'固定資料的混合 Train／Val／Test 在六個條件均較初始化改善；真實 2018 Val 僅 {val_positive}/6、真實 2019 Test 僅 {test_positive}/6 較初始化改善。主要改善來自合成情境，尚不能據此宣稱真實行情泛化或指定交易行為已改善。']
    lines += ['', '## 圖表與 reward 定義', '',
        '- `actual_training_reward_R_H_C_D.png` 是原始 SB3 訓練紀錄：含隨機探索、最近 100 個完整 episode 的平均。每列在當次梯度更新之前記錄；CSV 同時提供收集步數及實際使用策略的訓練步數。最後一列由 checkpoint 31 收集，尚未反映第 32 次更新後策略。',
        '- H/C/D 整體訓練 episode 比例是 50% 真實、50% 合成；logger 最後 100 個 episode 的移動視窗未必恰好各半。',
        '- `R/H/C/D_train_val_test.png` 使用固定路徑、確定性策略比較三個 split。R 僅真實資料；H/C/D 各 split 均為 50% 真實平均 reward + 50% 該組合成平均 reward。不是將不同數量的 episode 直接混平均。',
        '- `common_real_R/H/C/D_train_val_test.png` 只用各組完全相同的真實資料，可比較一般行情表現。R/H 的模型在同一情境 E0/E1 共用，因此每情境只畫一次。',
        '- `D_same_pool_update_gains.png` 在每次更新前後用同一組保留的訓練價格路徑評估，排除該次池抽樣差異；D 每兩次更新換池，不將跨池 gain 加總為學習幅度。',
        '- reward 是一個 21 日 episode 的日 reward 總和：`100 × ln(V21 / V0)`，再跨 episode 平均；不是日均 reward、簡單報酬百分比或年化績效。費率 0.0025、初始全現金，初次建倉也扣費。負 reward 向零改善也屬上升。',
        '- 原始 train reward 有探索噪音且池持續變動；其數值不能直接與確定性 val/test 的高低作過擬合判斷。固定 split 評估曲線可減少這個混淆。', '',
        '## 固定評估資料', '',
        '- Train：從 3,202 個 2005–2017 真實訓練 episode 等距選 64 個，未來交易日互不重疊；合成訓練從各 800 條中以固定 seed 20261115 選 64 條。是固定樣本評估，不是全訓練集平均。C/D 使用完全相同的真實及合成路徑。抽樣在評估任何 checkpoint 前鎖定。',
        '- Val：原先的 2018 真實 9 個非重疊 episode 及合成 200 條，沿用訓練時的確定性評估向量；所有 18 組初始與最終 checkpoint 重新評估並核對。',
        '- Test：本次依要求新增 2019 真實 9 個非重疊 episode，以及每條件先前已獨立生成的 200 條 test 路徑。真實資料的 50 日歷史與 21 日目標均在 2019 年。',
        '- 2019 真實評分日期：'+ '; '.join(f'{a}～{b}' for a,b in manifest['banks']['real_test']['targets'])+'。這是 episode 評估，不是全年連續持倉回測。',
        '- 合成 test 使用 2019 背景，但 C/D 條件仍包含事件實現值建立的事後情境標籤；合成測試改善不能視為真實市場事件交易改善。合成路徑共用背景時有相關性，不能當成獨立重複實驗。',
        '- 新增 test 曲線僅供診斷，不用來選 checkpoint、重新訓練或調 D。正式 2020/2022 真實事件尚未回測。2019 現已被檢視，後續若據此改動設定需另記為探索分析。', '',
        '## D：固定資料上的初始至最終改善', '',
        '| 情境 | 混合Train Δ | 混合Val Δ | 混合Test Δ | 真實Val Δ | 真實Test Δ | 最終真實Test D−C |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in pilot.CASES:
        v=final[(final['case']==case)&(final['group']=='D')].set_index('split')
        p=paired[(paired['case']==case)&(paired['update']==32)&(paired['split']=='test')].iloc[0]
        lines.append(f'| {case_label(case).replace("|", "／")} | {v.loc["train","matched_improvement"]:+.3f} | {v.loc["validation","matched_improvement"]:+.3f} | {v.loc["test","matched_improvement"]:+.3f} | {v.loc["validation","real_improvement"]:+.3f} | {v.loc["test","real_improvement"]:+.3f} | {p.real_D_minus_C:+.3f} |')
    lines += ['', '## 產出與可追溯性', '',
        '- `all_reward_curves.pdf`：全部圖的多頁 PDF。',
        '- `checkpoint_reward_curves.csv`：每個條件、組別、checkpoint、split 的真實、合成及固定比例混合平均值。',
        '- `rollout_training_rewards.csv`：原始訓練 reward 與時間對齊資訊；`D_training_trends.csv`：上表數據。',
        '- `initial_vs_final.csv`、`paired_D_vs_C.csv`：初末差值及同 checkpoint D/C 配對比較，沒有依 test 挑選 checkpoint。',
        '- `evaluations/*.npz`：逐 checkpoint、逐 episode reward；`evaluation_manifest.json`：固定選樣、日期、來源 SHA-256 與 checkpoint SHA-256；`audit.json`：完整性與來源未變動核對。',
        '- `source_snapshot.py` 保留最初評估程式；`reporting_source_snapshot.py` 保留修正報告排版與續跑狀態紀錄後的程式。兩者雜湊另記於 audit；評估計算相同。',
        '- 原有模型、訓練紀錄、資料銀行及先前報告均保留原狀。', '', '圖檔：', '']
    lines += [f'- [{name}]({out/name})' for name in artifacts]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def audit(out, manifest, started, artifacts):
    # All input checkpoints and training records must still match the pre-evaluation snapshot.
    for source, expected in manifest['source_files'].items():
        assert pilot.sha(source) == expected, source
    for name, metadata in manifest['banks'].items():
        assert pilot.sha(out/'evaluation_banks'/f'{name}.npy') == metadata['sha256']
    statuses=[json.loads((out/'evaluations'/f'{job}.json').read_text()) for job in manifest['jobs']]
    assert all(s['status']=='complete' and s['completed_checkpoints']==33 for s in statuses)
    points=pd.read_csv(out/'checkpoint_reward_curves.csv')
    assert len(points)==6*4*33*3 and np.isfinite(points[['real_reward','matched_reward']]).all().all()
    assert not points.duplicated(['case','group','update','split']).any()
    assert all(s['validation_reproduction_checks_this_run']==[0,32] for s in statuses)
    payload={'status':'passed','completed_utc':pilot.now(),'unique_policies':18,'checkpoints_per_policy':33,
        'evaluation_script_sha256':manifest['script_sha256'], 'reporting_script_sha256':pilot.sha(__file__),
        'checkpoint_evaluations':594,'condition_group_split_points':len(points),'source_files_unchanged':len(manifest['source_files']),
        'validation_reproduction_initial_and_final':True,'real_event_test_scored':False,
        'training_steps_added':0,'checkpoint_selection_performed':False,'elapsed_seconds':time.monotonic()-started,
        'artifact_sha256':{name:pilot.sha(out/name) for name in artifacts+['all_reward_curves.pdf','REPORT.md',
            'checkpoint_reward_curves.csv','rollout_training_rewards.csv','D_training_trends.csv','initial_vs_final.csv','paired_D_vs_C.csv']}}
    pilot.write_json(out/'audit.json',payload)
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--mode',choices=['run','worker','render'],default='run')
    parser.add_argument('--job')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='worker':
        worker(args.out,args.job)
        return
    if args.mode=='render':
        render(args.out)
        return
    started=time.monotonic()
    manifest=prepare(args.out)
    pilot.write_json(args.out/'status.json',{'status':'running','stage':'frozen_checkpoint_evaluation','updated_utc':pilot.now()})
    def one(name):
        with (args.out/'evaluations'/f'{name}.log').open('a') as log:
            command=[sys.executable,str(Path(__file__).resolve()),'--mode','worker','--out',str(args.out),'--job',name]
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env={**os.environ,'MPLBACKEND':'Agg'}).returncode
        if code:
            raise RuntimeError(f'{name} exited {code}; see evaluations/{name}.log')
        print(f'Completed {name}',flush=True)
    (args.out/'evaluations').mkdir(exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(one,name) for name in manifest['jobs']]
            for future in as_completed(futures):
                future.result()
        artifacts=render(args.out)
        result=audit(args.out,manifest,started,artifacts)
        pilot.write_json(args.out/'status.json',{**result,'status':'complete','completed_utc':pilot.now()})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as error:
        pilot.write_json(args.out/'status.json',{'status':'failed','error':str(error),'updated_utc':pilot.now()})
        raise


if __name__=='__main__':
    main()
