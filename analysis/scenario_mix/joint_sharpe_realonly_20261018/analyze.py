"""Audit, select by validation Sharpe, replay validation, and plot matched PPO runs.

Run from FinRL_Exp with its venv Python. No training or test evaluation is performed.
Commands: audit, evaluate [--groups ...], summarize. Original runs are read only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/scenario_mix/joint_pr30_pr10_real_only_sharpe_20261018"
REAL = ROOT / "results/scenario_mix/joint_real_only_seeds0_9_20261018_run01"
MIXED = ROOT / "results/scenario_mix/joint_pr30_pr10_b50_seeds0_9_20261018_run02"
PRICES = ROOT.parent / "TimeDiff/datasets/dow10/prices_2004_2025.csv"
GROUPS = ["REAL_ONLY", "PR30_B50", "PR10_B50"]
LABELS = {"REAL_ONLY": "Real-only", "PR30_B50": "PR30", "PR10_B50": "PR10"}
COLORS = {"REAL_ONLY": "#67727D", "PR30_B50": "#2878B5", "PR10_B50": "#D77A25"}
STYLES = {"REAL_ONLY": "--", "PR30_B50": "-", "PR10_B50": "-."}
STEPS = list(range(25000, 500001, 25000))
ACTUAL = list(range(26624, 532481, 26624))
VALID = ["2018-01-01", "2019-12-31"]
SEEDS = list(range(10))

# Match the invocation-only workaround used during both training batches.
sys.modules.setdefault("quantstats", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-joint-sharpe-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/finrl-joint-sharpe-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def run_path(group, seed):
    return (REAL if group == "REAL_ONLY" else MIXED) / f"{group}_seed{seed}"


def read_validation(group, seed):
    path = run_path(group, seed)
    cfg = json.loads((path / "config.json").read_text())
    reference = json.loads((MIXED / "PR30_B50_seed0/config.json").read_text())
    for key in ["train", "valid", "timesteps", "eval_freq", "commission", "k_max", "log_std_init", "ent_coef", "rollout_steps"]:
        assert cfg[key] == reference[key], (path, key, cfg[key], reference[key])
    effective_fee = lambda c: c["commission"] if c["train_commission"] is None else c["train_commission"]
    assert effective_fee(cfg) == effective_fee(reference) == .0025
    assert cfg["p"] == (0 if group == "REAL_ONLY" else .5)
    assert cfg["seed"] == seed and cfg["group"] == group
    v = pd.read_csv(path / "valid.csv")
    assert v.timesteps.tolist() == STEPS, path
    assert v.actual_timesteps.tolist() == ACTUAL, path
    assert np.isfinite(v.to_numpy()).all(), path
    assert np.allclose(v.valid_reward, 100 * np.log(v.valid_final_value), rtol=1e-10)
    return v.assign(group=group, seed=seed)


def selected_row(group, seed):
    v = read_validation(group, seed).sort_values("timesteps")
    return v.loc[v.valid_sharpe.idxmax()].to_dict()


def audit():
    status = json.loads((REAL / "status.json").read_text())
    assert set(status) == {f"REAL_ONLY_seed{s}" for s in SEEDS}
    assert all(s["state"] == "succeeded" and s["returncode"] == 0 for s in status.values())
    audit_rows = []
    for seed in SEEDS:
        v = read_validation("REAL_ONLY", seed)
        p = run_path("REAL_ONLY", seed)
        ep = pd.read_csv(p / "episodes.csv")
        assert len(ep) > 0 and (ep.source == "real").all() and (ep.scenario == -1).all()
        assert (ep.length == ep.k + 21).all() and ep.k.between(0, 21).all()
        assert np.isfinite(ep[["reward", "length", "timesteps"]].to_numpy()).all()
        files = sorted((p / "ckpt").glob("step_*.zip"))
        assert len(files) == 20
        for nominal, actual in zip(STEPS, ACTUAL):
            with zipfile.ZipFile(p / "ckpt" / f"step_{nominal:06d}.zip") as archive:
                assert archive.testzip() is None
                assert {"data", "policy.pth", "policy.optimizer.pth"} <= set(archive.namelist())
                data = json.loads(archive.read("data"))
                assert data["num_timesteps"] == actual and data["seed"] == seed
                assert data["n_steps"] == 2048 and data["n_envs"] == 1
        log = (REAL / "logs" / f"REAL_ONLY_seed{seed}.log").read_text()
        assert log.strip().endswith("done") and "Traceback" not in log
        audit_rows.append({"seed": seed, "checkpoints": len(files), "validation_rows": len(v), "episode_rows": len(ep), "all_episodes_real": True, "last_nominal": int(v.timesteps.iloc[-1]), "last_actual": int(v.actual_timesteps.iloc[-1])})
    launch = json.loads((REAL / "runtime/launch.json").read_text())
    for name, expected in launch["input_sha256"].items():
        assert sha(name) == expected, name
    for name, expected in launch["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    prior_launch = json.loads((MIXED / "runtime/launch.json").read_text())
    for name, expected in prior_launch["input_sha256"].items():
        assert sha((ROOT / name).resolve()) == expected, name
    reference_commit = prior_launch["git_commit"]
    for name, expected in launch["source_sha256"].items():
        reference_source = subprocess.run(["git", "show", f"{reference_commit}:{name}"], cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(reference_source).hexdigest() == expected, ("Training source differs from paired batch commit", name)
    start = datetime.fromisoformat(launch["started_at"])
    completed = max(datetime.fromisoformat(x["updated_at"]) for x in status.values())
    summary = {"status": "completed_and_verified", "checked_at": datetime.now(timezone.utc).isoformat(), "started_at": start.isoformat(), "completed_at": completed.isoformat(), "elapsed_seconds": (completed-start).total_seconds(), "runs_succeeded": 10, "runs_failed": 0, "workers": 5, "checkpoints_total": 200, "validation_rows_total": 200, "all_training_episodes_real": True, "test_evaluation_performed": False, "input_and_training_source_hashes_unchanged": True, "training_source_matches_paired_batch_commit": reference_commit, "effective_training_commission": .0025, "config_note": "Real-only train_commission=null inherits commission=0.0025; the mixed batch explicitly sets train_commission=0.0025. Effective training and validation fees are identical.", "runs": audit_rows}
    write_json(REAL / "completion_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def metrics(daily):
    values = daily.value.to_numpy(float)
    log_returns = np.diff(np.log(values))
    simple_returns = np.expm1(log_returns)
    drawdown = values / np.maximum.accumulate(values) - 1
    annual = np.expm1(log_returns.mean() * 252)
    q05 = np.quantile(simple_returns, .05)
    tail = simple_returns[simple_returns <= q05]
    result = {"return_pct": (values[-1] - 1) * 100, "annual_return_pct": annual * 100,
              "sharpe": log_returns.mean() / log_returns.std(ddof=1) * np.sqrt(252),
              "calmar": annual / abs(drawdown.min()), "mdd_pct": -drawdown.min() * 100,
              "es95_daily_pct": -tail.mean() * 100, "var95_daily_pct": -q05 * 100,
              "stock_weight_pct": (1-daily.cash_weight.mean()) * 100}
    if "turnover" in daily:
        result["annual_turnover"] = daily.turnover.iloc[1:].sum() / len(log_returns) * 252
        result["fee_factor_drag_pct"] = (1 - daily.trf_mu.prod()) * 100
        weights = daily[[c for c in daily if c.startswith("weight_") and c != "weight_CASH"]].to_numpy()
        result["mean_largest_stock_weight_pct"] = weights.max(axis=1).mean() * 100
        result["action_bound_fraction_pct"] = daily.action_bound_fraction.iloc[1:].mean() * 100
    return {k: float(v) for k, v in result.items()}


def validation_rollout(model, market, commission):
    from finrl.experiments import scenario_mix as S
    s, e = market.idx(VALID[0]), market.last_idx(VALID[1])
    rows = slice(s - 50, e + 1)
    frame = S.episode_frame(market.dates[rows], market.close[rows], market.high[rows], market.low[rows], market.tickers)
    inner = S._InnerEpisode(frame, len(market.tickers), commission, 50)
    obs, _ = inner.reset()
    obs = S.normalize_obs(obs)
    mus, bounds = [], []
    done = False
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        bounds.append(float(np.isclose(np.abs(action), 1., atol=1e-6).mean()))
        obs, _, done, _, info = inner.step(action)
        mus.append(float(info["trf_mu"]))
        obs = S.normalize_obs(obs)
    env = inner.env
    n = len(market.tickers)
    entry = 1 - commission*n/(n+1)
    values = np.asarray(env._asset_memory["final"], float) / env._initial_amount * entry
    acts = np.asarray(env._actions_memory, float)
    finals = np.asarray(env._final_weights, float)
    daily = pd.DataFrame({"date": market.dates[s-1:e+1], "value": values, "cash_weight": finals[:,0],
                          "turnover": np.r_[n/(n+1), np.abs(acts[1:]-finals[:-1]).sum(axis=1)/2],
                          "trf_mu": np.r_[entry, mus], "action_bound_fraction": np.r_[np.nan, bounds]})
    for i, name in enumerate(["CASH", *market.tickers]):
        daily[f"weight_{name}"] = finals[:,i]
        daily[f"target_{name}"] = acts[:,i]
    return daily


def evaluate(groups):
    import torch
    from stable_baselines3 import PPO
    from finrl.experiments import scenario_mix as S
    from finrl.experiments.scenario_mix_common import TICKERS
    torch.set_num_threads(1)
    market = S.MarketData(PRICES, TICKERS)
    (OUT / "daily").mkdir(exist_ok=True)
    (OUT / "evaluations").mkdir(exist_ok=True)
    for group in groups:
        for seed in SEEDS:
            selection = selected_row(group, seed)
            checkpoint = run_path(group, seed) / "ckpt" / f"step_{int(selection['timesteps']):06d}.zip"
            digest = sha(checkpoint)
            meta_file = OUT / "evaluations" / f"{group}_seed{seed}.json"
            daily_file = OUT / "daily" / f"{group}_seed{seed}_validation.csv.gz"
            if meta_file.exists() and daily_file.exists():
                saved = json.loads(meta_file.read_text())
                assert saved["checkpoint_sha256"] == digest
                assert saved["daily_sha256"] == sha(daily_file)
                print(f"verified cached validation: {group} seed {seed}", flush=True)
                continue
            model = PPO.load(checkpoint, device="cpu")
            assert model.num_timesteps == selection["actual_timesteps"] and model.seed == seed
            daily = validation_rollout(model, market, .0025)
            actual = metrics(daily)
            expected = {"return_pct": (selection["valid_final_value"]-1)*100, "sharpe": selection["valid_sharpe"], "calmar": selection["valid_calmar"], "stock_weight_pct": selection["valid_mean_stock_weight"]*100}
            deviations = {key: actual[key]-value for key,value in expected.items()}
            assert all(np.isclose(actual[k], value, atol=1e-6, rtol=1e-6) for k,value in expected.items()), deviations
            if seed == 0:
                reference = S.run_continuous(lambda obs: model.predict(obs, deterministic=True)[0], market, *VALID, .0025)
                for column in ["value", "cash_weight", "turnover"]:
                    assert np.allclose(reference[column], daily[column], atol=1e-10, rtol=1e-10), column
            daily.to_csv(daily_file, index=False, compression={"method":"gzip", "mtime":0})
            write_json(meta_file, {"group":group, "seed":seed, "selection":selection, "checkpoint":str(checkpoint.relative_to(ROOT)), "checkpoint_sha256":digest,
                                  "daily_file":str(daily_file.relative_to(ROOT)), "daily_sha256":sha(daily_file), "validation_only":True,
                                  "metrics":actual, "replay_metric_deviations":deviations, "matches_original_rollout_checked_for_seed0":seed==0})
            print(f"validated {group} seed {seed}: step={int(selection['timesteps'])}, Sharpe={actual['sharpe']:.6f}, return={actual['return_pct']:.2f}%", flush=True)
    baseline = S.buy_and_hold(market, *VALID, .0025, include_cash=True)
    entry = 1 - .0025*10/11
    baseline["cash_weight"] = entry/11/baseline.value
    baseline.to_csv(OUT / "buy_hold_validation.csv", index=False)
    write_json(OUT / "buy_hold_metrics.json", metrics(baseline))


def collect():
    frames, selected, episodes, provenance = [], [], [], []
    for group in GROUPS:
        for seed in SEEDS:
            v = read_validation(group, seed)
            frames.append(v)
            meta_path = OUT / "evaluations" / f"{group}_seed{seed}.json"
            meta = json.loads(meta_path.read_text())
            chosen = selected_row(group, seed)
            assert chosen["timesteps"] == meta["selection"]["timesteps"]
            assert sha(ROOT / meta["checkpoint"]) == meta["checkpoint_sha256"]
            daily = pd.read_csv(ROOT / meta["daily_file"], parse_dates=["date"])
            measured = metrics(daily)
            assert all(np.isclose(measured[k], val, rtol=1e-8, atol=1e-8) for k,val in meta["metrics"].items())
            selected.append({"group":group, "seed":seed, "nominal_timesteps":int(chosen["timesteps"]), "actual_timesteps":int(chosen["actual_timesteps"]), "checkpoint":meta["checkpoint"], "checkpoint_sha256":meta["checkpoint_sha256"], **measured})
            ep = pd.read_csv(run_path(group, seed) / "episodes.csv")
            syn = ep[ep.source == "synthetic"]
            episodes.append({"group":group, "seed":seed, "episodes":len(ep), "synthetic_episodes":len(syn), "synthetic_pct":len(syn)/len(ep)*100, "tail_pool_episodes":int((ep.scenario==1).sum()), "tail_pool_all_pct":float((ep.scenario==1).mean()*100), "tail_pool_synthetic_pct":float((syn.scenario==1).mean()*100) if len(syn) else 0., "mean_length":float(ep.length.mean())})
            provenance.extend({"file":str(p.relative_to(ROOT)), "sha256":sha(p)} for p in [run_path(group, seed)/"config.json", run_path(group, seed)/"valid.csv", run_path(group, seed)/"episodes.csv"])
    v = pd.concat(frames, ignore_index=True)
    assert len(v)==600 and not v.duplicated(["group","seed","timesteps"]).any()
    baseline = pd.read_csv(OUT/"buy_hold_validation.csv")
    n = len(baseline)-1
    entry = float(baseline.value.iloc[0])
    v["return_pct"] = (v.valid_final_value-1)*100
    v["sharpe"] = v.valid_sharpe
    v["calmar"] = v.valid_calmar
    v["mdd_pct"] = np.expm1(np.log(v.valid_final_value/entry)*252/n)/v.valid_calmar*100
    v["stock_weight_pct"] = v.valid_mean_stock_weight*100
    for row in selected:
        inferred = v[(v.group==row["group"]) & (v.seed==row["seed"]) & (v.timesteps==row["nominal_timesteps"])].mdd_pct.iloc[0]
        assert np.isclose(inferred, row["mdd_pct"], rtol=1e-8, atol=1e-8)
    return v, pd.DataFrame(selected), pd.DataFrame(episodes), provenance


def summarize():
    v, selected, episodes, provenance = collect()
    final = v[v.timesteps==500000]
    measure_columns = ["return_pct","sharpe","calmar","mdd_pct","es95_daily_pct","stock_weight_pct","annual_turnover","fee_factor_drag_pct","mean_largest_stock_weight_pct","action_bound_fraction_pct"]
    summary = []
    for group in GROUPS:
        z = selected[selected.group==group]
        row = {"group":group, "n_seeds":len(z), "selected_step_median":float(z.nominal_timesteps.median())}
        for column in measure_columns:
            row.update({f"{column}_mean":float(z[column].mean()), f"{column}_sd":float(z[column].std(ddof=1)), f"{column}_median":float(z[column].median())})
        summary.append(row)
    pairs=[]
    rng=np.random.default_rng(20261018)
    indices=rng.integers(0,10,size=(200000,10))
    for treatment, control in [("PR30_B50","REAL_ONLY"),("PR10_B50","REAL_ONLY"),("PR10_B50","PR30_B50")]:
        a=selected[selected.group==treatment].set_index("seed").sort_index()
        b=selected[selected.group==control].set_index("seed").sort_index()
        for column in ["return_pct","sharpe","calmar","mdd_pct","es95_daily_pct"]:
            differences=(a[column]-b[column]).to_numpy()
            ci=np.quantile(differences[indices].mean(axis=1),[.025,.975])
            better=differences<0 if column in ["mdd_pct","es95_daily_pct"] else differences>0
            pairs.append({"treatment":treatment,"control":control,"metric":column,"mean_difference":float(differences.mean()),"ci95_low":float(ci[0]),"ci95_high":float(ci[1]),"treatment_better_seeds":int(better.sum()),"n_paired_seeds":10})
    v.to_csv(OUT/"all_validation_checkpoints.csv",index=False)
    selected.to_csv(OUT/"selected_checkpoints.csv",index=False)
    pd.DataFrame(summary).to_csv(OUT/"selected_summary.csv",index=False)
    final.to_csv(OUT/"fixed_final_checkpoints.csv",index=False)
    final.groupby("group")[["return_pct","sharpe","calmar","mdd_pct","stock_weight_pct"]].agg(["mean","std","median"]).to_csv(OUT/"fixed_final_summary.csv")
    episodes.to_csv(OUT/"episode_sampling.csv",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"paired_seed_differences.csv",index=False)
    sensitivity = selected[["group", "seed", "nominal_timesteps", "sharpe", "return_pct", "mdd_pct"]].merge(final[["group", "seed", "sharpe", "return_pct", "mdd_pct"]], on=["group", "seed"], suffixes=("_selected", "_final"))
    for metric in ["sharpe", "return_pct", "mdd_pct"]:
        sensitivity[f"{metric}_final_minus_selected"] = sensitivity[f"{metric}_final"] - sensitivity[f"{metric}_selected"]
    sensitivity.to_csv(OUT/"checkpoint_sensitivity.csv",index=False)
    period_rows, weight_rows = [], []
    periods = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31")]
    for year in [2018, 2019]:
        for quarter in range(1, 5):
            period = pd.Period(f"{year}Q{quarter}", freq="Q")
            periods.append((str(period), str(period.start_time.date()), str(period.end_time.date())))
    series = [(row.group, int(row.seed), OUT/"daily"/f"{row.group}_seed{int(row.seed)}_validation.csv.gz") for row in selected.itertuples()]
    series.append(("BUY_HOLD", -1, OUT/"buy_hold_validation.csv"))
    for group, seed, path in series:
        daily = pd.read_csv(path, parse_dates=["date"])
        if group != "BUY_HOLD":
            for col in [c for c in daily if c.startswith("weight_")]:
                weight_rows.append({"group":group,"seed":seed,"asset":col.removeprefix("weight_"),"mean_weight_pct":float(daily[col].mean()*100),"max_weight_pct":float(daily[col].max()*100)})
        for label, start, end in periods:
            left = daily.date.searchsorted(pd.Timestamp(start))
            right = daily.date.searchsorted(pd.Timestamp(end), side="right")
            segment = daily.iloc[left-1:right].copy()
            base = 1. if start == VALID[0] else float(segment.value.iloc[0])
            segment["value"] = segment.value/base
            m = metrics(segment)
            period_rows.append({"group":group,"seed":seed,"period":label,"return_days":len(segment)-1,**{k:m[k] for k in ["return_pct","sharpe","mdd_pct"]}})
    periods_df = pd.DataFrame(period_rows)
    periods_df.to_csv(OUT/"validation_subperiods.csv",index=False)
    periods_df.groupby(["group","period"])[["return_pct","sharpe","mdd_pct"]].agg(["mean","std","median"]).to_csv(OUT/"validation_subperiod_summary.csv")
    pd.DataFrame(weight_rows).to_csv(OUT/"portfolio_weights_summary.csv",index=False)
    write_json(OUT/"source_manifest.json", {"files":provenance,"prices":{"file":str(PRICES),"sha256":sha(PRICES)},"analysis_script_sha256":sha(__file__)})
    analysis = {"status":"complete","generated_at":datetime.now(timezone.utc).isoformat(),"groups":GROUPS,"seeds":SEEDS,"validation_period":VALID,"n_validation_return_days":len(pd.read_csv(OUT/"buy_hold_validation.csv"))-1,"test_evaluation_performed":False,"effective_training_and_validation_commission":.0025,
       "selection_rule":json.loads((OUT/"selection_rule.json").read_text()),"summary":summary,"paired_differences":pairs,"buy_hold":json.loads((OUT/"buy_hold_metrics.json").read_text()),
       "definitions":{"return_pct":"Cumulative net validation return relative to pre-entry capital; percent, not annualized.","sharpe":"mean(daily log return)/sample SD * sqrt(252), zero risk-free rate.","mdd_pct":"Positive maximum drawdown magnitude; computed directly from each selected checkpoint daily path.","es95_daily_pct":"Positive loss magnitude: minus average simple daily return at or below its empirical 5th percentile.","stock_weight_pct":"Average total stock allocation across validation, including initial allocation.","annual_turnover":"252 * mean daily half-L1 weight turnover, excluding initial build-up.","fee_factor_drag_pct":"100*(1-product(TRF multipliers including entry)); multiplicative fee drag holding actions fixed, not a separately optimized no-fee strategy.","action_bound_fraction_pct":"Mean proportion of clipped action coordinates at +/-1 across validation decision days.","paired_ci":"Percentile bootstrap of 10 seed-paired mean differences, 200000 resamples, RNG seed 20261018; training-seed variability only.","curve_band":"Across-seed 25th-75th percentiles, not a confidence interval.","subperiods":"Carry existing positions across year/quarter boundaries; no policy reset. Periods beginning 2018-01-01 include initial entry cost; later periods use the preceding trading-day value as base. Subperiods are descriptive, overlapping diagnostics, not extra independent trials."},
       "limitations":["Checkpoint selection and reported main scores use the same validation period, so scores are selection-optimistic.","TimeDiff model selection also used 2018-2019 validation.","Only one market validation period and one fixed pair of generated datasets are represented.","Real-only and mixture runs share seed labels and hyperparameters, but their random episode sequences differ when p changes.","No test results are included."]}
    plot(v, selected, pd.DataFrame(pairs))
    write_json(OUT/"analysis.json", analysis)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def plot(v, selected, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.dates as mdates
    font=Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"figure.facecolor":"white","axes.facecolor":"white","font.size":11,"axes.labelsize":11,"axes.titlesize":13,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#E5E8EB","grid.alpha":.8,"grid.linewidth":.6,"axes.axisbelow":True,"savefig.facecolor":"white","svg.fonttype":"path"})
    figures=OUT/"figures"; figures.mkdir(exist_ok=True)
    baseline=json.loads((OUT/"buy_hold_metrics.json").read_text())

    def save(fig,name,footnote):
        fig.text(.02,.015,footnote,ha="left",va="bottom",fontsize=9,color="#555D65")
        fig.savefig(figures/f"{name}.png",dpi=180,bbox_inches="tight")
        fig.savefig(figures/f"{name}.svg",bbox_inches="tight")
        plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    settings=[("return_pct","累積報酬（%）"),("sharpe","Sharpe"),("mdd_pct","最大回撤幅度（%，越低越好）"),("calmar","Calmar")]
    for ax,(metric,label) in zip(axes.flat,settings):
        for x,group in enumerate(GROUPS):
            values=selected[selected.group==group].sort_values("seed")[metric].to_numpy()
            bp=ax.boxplot(values,positions=[x],widths=.42,patch_artist=True,showfliers=False,manage_ticks=False,medianprops={"color":"#222222","linewidth":1.4},whiskerprops={"color":COLORS[group]},capprops={"color":COLORS[group]})
            bp["boxes"][0].set(facecolor=COLORS[group],alpha=.18,edgecolor=COLORS[group])
            ax.scatter(x+np.linspace(-.12,.12,10),values,c=COLORS[group],s=30,alpha=.85,zorder=3)
            ax.scatter([x],[values.mean()],marker="D",s=60,c=COLORS[group],edgecolors="white",linewidths=1,zorder=4)
        ax.axhline(baseline[metric],color="#252A30",linestyle=":",linewidth=1.4,label="買入持有")
        ax.set(xticks=range(3),xticklabels=[LABELS[g] for g in GROUPS],ylabel=label,title=label,xlim=(-.5,2.5))
        ax.grid(axis="x",visible=False)
    axes[0,0].legend(frameon=False,loc="upper left")
    fig.suptitle("以 validation Sharpe 選模後的表現",fontsize=19,y=.99)
    fig.subplots_adjust(top=.91,bottom=.12,hspace=.32,wspace=.25)
    save(fig,"01_selected_metrics","2018–2019 validation｜每組 10 seeds；圓點＝seed，菱形＝平均，盒體＝25–75%／中位數。\n同一 validation 用於選模與評分；本圖不是獨立 test 結果。")

    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,(metric,label) in zip(axes.flat,[settings[0],settings[1],settings[2],("stock_weight_pct","平均股票曝險（%）")]):
        for group in GROUPS:
            pivot=v[v.group==group].pivot(index="timesteps",columns="seed",values=metric)
            x=pivot.index.to_numpy()/1000
            ax.plot(x,pivot.mean(axis=1),color=COLORS[group],linestyle=STYLES[group],linewidth=2,label=LABELS[group])
            ax.fill_between(x,pivot.quantile(.25,axis=1),pivot.quantile(.75,axis=1),color=COLORS[group],alpha=.12,linewidth=0)
        ax.axhline(baseline[metric],color="#252A30",linestyle=":",linewidth=1)
        ax.set(ylabel=label,title=label,xlim=(25,500),xticks=[25,100,200,300,400,500])
    axes[0,0].legend(frameon=False,ncol=3,fontsize=10)
    for ax in axes[-1]: ax.set_xlabel("名義訓練步數（千步）")
    fig.suptitle("訓練過程的 validation 表現",fontsize=19,y=.99)
    fig.subplots_adjust(top=.91,bottom=.12,hspace=.32,wspace=.25)
    save(fig,"02_training_curves","線＝10 seeds 平均；色帶＝25–75% seeds 範圍，非信賴區間；黑點線＝買入持有。\n名義 500k 對應實際 532,480 步。每個點均評估相同的 2018–2019 validation。")

    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for group in GROUPS:
        paths=[pd.read_csv(OUT/"daily"/f"{group}_seed{s}_validation.csv.gz",parse_dates=["date"]) for s in SEEDS]
        dates=paths[0].date
        values=np.column_stack([d.value for d in paths])
        dd=(values/np.maximum.accumulate(values,axis=0)-1)*100
        for ax,array in zip(axes,[values,dd]):
            ax.plot(dates,np.median(array,axis=1),color=COLORS[group],linestyle=STYLES[group],linewidth=2,label=LABELS[group])
            ax.fill_between(dates,np.quantile(array,.25,axis=1),np.quantile(array,.75,axis=1),color=COLORS[group],alpha=.12,linewidth=0)
    bh=pd.read_csv(OUT/"buy_hold_validation.csv",parse_dates=["date"])
    axes[0].plot(bh.date,bh.value,color="#252A30",linestyle=":",linewidth=1.5,label="買入持有")
    axes[1].plot(bh.date,(bh.value/bh.value.cummax()-1)*100,color="#252A30",linestyle=":",linewidth=1.5)
    axes[0].set_ylabel("淨值（初始資本＝1）"); axes[1].set_ylabel("當日回撤（%）")
    axes[0].legend(frameon=False,ncol=4)
    axes[1].set_xticks(pd.to_datetime(["2017-12-29","2018-07-01","2019-01-01","2019-07-01","2019-12-31"]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].set_xlim(bh.date.iloc[0],bh.date.iloc[-1])
    fig.suptitle("Sharpe 選模後：validation 淨值與回撤路徑",fontsize=19,y=.99)
    fig.subplots_adjust(top=.91,bottom=.12,hspace=.14)
    save(fig,"03_validation_paths","線＝每日跨 seeds 中位數；色帶＝25–75%；回撤先逐 seed 計算再彙整。\n起始點為 2017-12-29 收盤／扣除建倉成本後；交易評估期間為 2018–2019。")

    fig,axes=plt.subplots(1,3,figsize=(13,5))
    comparisons=[("PR30_B50","REAL_ONLY"),("PR10_B50","REAL_ONLY"),("PR10_B50","PR30_B50")]
    for ax,(metric,label) in zip(axes,[("return_pct","累積報酬差（百分點）"),("sharpe","Sharpe 差"),("mdd_pct","回撤幅度差（百分點）")]):
        for y,(treatment,control) in enumerate(comparisons):
            row=pairs[(pairs.treatment==treatment)&(pairs.control==control)&(pairs.metric==metric)].iloc[0]
            a=selected[selected.group==treatment].set_index("seed")[metric]
            b=selected[selected.group==control].set_index("seed")[metric]
            diff=(a-b).sort_index().to_numpy()
            ax.scatter(diff,y+np.linspace(-.12,.12,10),color=COLORS[treatment],s=23,alpha=.5)
            ax.errorbar(row.mean_difference,y,xerr=[[row.mean_difference-row.ci95_low],[row.ci95_high-row.mean_difference]],fmt="D",color="#252A30",markersize=6,capsize=4,linewidth=1.7,zorder=5)
        ax.axvline(0,color="#939BA2",linewidth=1)
        ax.set(yticks=range(3),yticklabels=[f"{LABELS[a]} − {LABELS[b]}" for a,b in comparisons],xlabel=label,ylim=(2.5,-.5),title="負值較好" if metric=="mdd_pct" else "正值較好")
        ax.grid(axis="y",visible=False)
    fig.suptitle("相同 seed 的配對差異",fontsize=19,y=.99)
    fig.subplots_adjust(top=.82,bottom=.22,wspace=.55)
    save(fig,"04_paired_differences","圓點＝seed 差值；菱形與線段＝平均差與配對 bootstrap 95% 區間（200,000 次）。\n區間只反映這 10 seeds 的變異，不涵蓋市場時期、生成資料種子或選模偏差。")

    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    for ax,(metric,label) in zip(axes,[("stock_weight_pct","平均股票曝險（%）"),("mean_largest_stock_weight_pct","平均最大單股權重（%）"),("es95_daily_pct","日報酬 ES95 損失幅度（%）")]):
        for x,group in enumerate(GROUPS):
            z=selected[selected.group==group][metric]
            ax.scatter(x+np.linspace(-.12,.12,10),z,color=COLORS[group],s=35,alpha=.75)
            ax.scatter(x,z.mean(),marker="D",s=65,color=COLORS[group],edgecolor="white",zorder=4)
        if metric in baseline: ax.axhline(baseline[metric],color="#252A30",linestyle=":",linewidth=1.3)
        ax.set(xticks=range(3),xticklabels=[LABELS[g] for g in GROUPS],ylabel=label,title=label,xlim=(-.5,2.5))
        ax.grid(axis="x",visible=False)
    fig.suptitle("Sharpe 選模後的曝險與尾部損失",fontsize=19,y=.99)
    fig.subplots_adjust(top=.82,bottom=.20,wspace=.35)
    save(fig,"05_exposure_and_tail","每個圓點＝一個 seed，菱形＝平均；黑點線＝買入持有（適用指標）。\nES95＝validation 最差 5% 日報酬的平均損失幅度；本圖沒有使用 test。")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["audit","evaluate","summarize"])
    parser.add_argument("--groups",nargs="+",choices=GROUPS,default=GROUPS)
    args=parser.parse_args()
    rule=json.loads((OUT/"selection_rule.json").read_text())
    assert rule["metric_column"]=="valid_sharpe" and rule["test_evaluation"] is False
    if args.command=="audit": audit()
    elif args.command=="evaluate": evaluate(args.groups)
    else: summarize()


if __name__=="__main__":
    main()
