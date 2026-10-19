"""Build auditable tables from frozen, uniformly evaluated checkpoints."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.scenario_mix.comprehensive_20261019 import evaluate as E
import numpy as np
import pandas as pd

METRICS=["return_pct","cagr_net_pct","sharpe","sortino","calmar","mdd_pct","volatility_pct","es95_daily_pct","var95_daily_pct",
         "annual_turnover","fee_factor_drag_pct","stock_weight_pct","mean_largest_stock_weight_pct","max_single_stock_weight_pct","concentrated_over80_pct","action_bound_fraction_pct",
         "worst_21d_return_pct","var95_21d_pct","es95_21d_pct"]


def horizon_metrics(values):
    rolling=values[21:]/values[:-21]-1
    assert len(rolling)>0 and np.isfinite(rolling).all()
    q=np.quantile(rolling,.05)
    return {"worst_21d_return_pct":float(rolling.min()*100),"var95_21d_pct":float(-q*100),"es95_21d_pct":float(-rolling[rolling<=q].mean()*100)}


def audit_training():
    status=json.loads((E.NEW/"status.json").read_text())
    expected={f"{g}_seed{s}" for g in ["REAL_REPLAY_PR30_B50","REAL_REPLAY_PR10_B50","FULL_B50"] for s in range(10)}
    assert set(status)==expected and all(v["state"]=="succeeded" and v["returncode"]==0 for v in status.values())
    launch=json.loads((E.OUT/"runtime/launch.json").read_text())
    for p,digest in launch["input_sha256"].items():assert E.sha(p)==digest,p
    for p,digest in launch["source_sha256"].items():assert E.sha(ROOT/p)==digest,p
    rows=[]
    for rec in E.records(["REAL_REPLAY_PR30_B50","REAL_REPLAY_PR10_B50","FULL_B50"]):
        cfg,v,inventory=E.checkpoint_inventory(rec)
        episodes=pd.read_csv(Path(rec["run"])/"episodes.csv")
        allowed={"real","real_replay"} if rec["group"].startswith("REAL_REPLAY") else {"real","synthetic"}
        assert set(episodes.source)==allowed
        assert (episodes.length==episodes.k+21).all() and episodes.k.between(0,21).all()
        assert np.isfinite(episodes[["timesteps","reward","length"]]).all().all()
        if rec["group"]=="FULL_B50":assert (episodes.loc[episodes.source=="synthetic","scenario"]==0).all()
        if rec["group"].startswith("REAL_REPLAY"):
            matched_group=rec["group"].replace("REAL_REPLAY_", "")
            matched=pd.read_csv(E.MIXED/f"{matched_group}_seed{rec['seed']}"/"episodes.csv")
            for col in ["timesteps","length","scenario","k"]:
                np.testing.assert_array_equal(episodes[col],matched[col])
            np.testing.assert_array_equal(episodes.source.replace({"real_replay":"synthetic"}),matched.source)
        log=(E.NEW/"logs"/f"{rec['id']}.log").read_text()
        assert log.strip().endswith("done") and "Traceback" not in log
        rows.append({"group":rec["group"],"seed":rec["seed"],"checkpoints":20,"validation_rows":20,"episodes":len(episodes),
                     "source_counts":episodes.source.value_counts().to_dict(),"matched_episode_schedule_verified":rec["group"].startswith("REAL_REPLAY"),
                     "last_nominal":int(v.timesteps.iloc[-1]),"last_actual":int(v.actual_timesteps.iloc[-1])})
    start=datetime.fromisoformat(launch["started_at"])
    end=max(datetime.fromisoformat(v["updated_at"]) for v in status.values())
    result={"status":"completed_and_verified","checked_at":datetime.now(timezone.utc).isoformat(),"started_at":start.isoformat(),"completed_at":end.isoformat(),
            "elapsed_seconds":(end-start).total_seconds(),"runs":30,"workers":5,"checkpoints":600,"validation_rows":600,"all_sources_verified":True,"runs_detail":rows}
    E.write(E.NEW/"completion_summary.json",result)
    return result


def group_summary(frame):
    result=[]
    for (group,split),z in frame.groupby(["group","split"],sort=False):
        row={"group":group,"split":split,"n_seeds":len(z),"selected_nominal_median":float(z.nominal_timesteps.median())}
        for metric in METRICS:
            row.update({f"{metric}_mean":float(z[metric].mean()),f"{metric}_sd":float(z[metric].std(ddof=1)),f"{metric}_median":float(z[metric].median()),f"{metric}_min":float(z[metric].min()),f"{metric}_max":float(z[metric].max())})
        result.append(row)
    return pd.DataFrame(result)


def benchmarks():
    from finrl.experiments import scenario_mix as S
    m=S.MarketData(E.PRICES,E.tickers());rows=[]
    destination=E.OUT/"benchmarks";destination.mkdir(exist_ok=True)
    for split,(start,end) in E.PERIODS.items():
        s,e=m.idx(start),m.last_idx(end)
        rel=m.close[s-1:e+1]/m.close[s-1]
        for include_cash in [True,False]:
            n=len(m.tickers);w=1/(n+1) if include_cash else 1/n
            initial_cash=1-w*n;entry=1-.0025*w*n
            individual=np.column_stack([np.full(len(rel),initial_cash),w*rel])
            raw_value=individual.sum(axis=1);weights=individual/raw_value[:,None]
            d=pd.DataFrame({"date":m.dates[s-1:e+1],"value":raw_value*entry,"cash_weight":weights[:,0],
                "turnover":np.r_[w*n,np.zeros(len(rel)-1)],"trf_mu":np.r_[entry,np.ones(len(rel)-1)],"action_bound_fraction":np.zeros(len(rel))})
            for i,t in enumerate(["CASH",*m.tickers]):d[f"weight_{t}"]=weights[:,i]
            expected=S.buy_and_hold(m,start,end,.0025,include_cash=include_cash)
            np.testing.assert_allclose(d.value,expected.value,atol=1e-12,rtol=1e-12)
            name="BUY_HOLD_CASH" if include_cash else "BUY_HOLD_STOCKS"
            d.to_csv(destination/f"{name}_{split}.csv.gz",index=False,compression={"method":"gzip","mtime":0})
            rows.append({"group":name,"split":split,**E.metrics(d),**horizon_metrics(d.value.to_numpy(float))})
    df=pd.DataFrame(rows);df.to_csv(E.OUT/"benchmark_metrics.csv",index=False)
    return df


def episode_tables():
    summaries=[];curves=[]
    for rec in E.records():
        ep=pd.read_csv(Path(rec["run"])/"episodes.csv")
        assert ep.source.notna().all()
        assert set(ep.source)<={"real","synthetic","real_replay"},rec
        ep["nominal_timesteps"]=np.ceil(ep.timesteps/26624).astype(int)*25000
        ep["reward_per_day"]=ep.reward/ep.length
        assert ep.nominal_timesteps.between(25000,500000).all()
        table=ep.groupby("nominal_timesteps").agg(episodes=("reward","size"),reward_mean=("reward","mean"),reward_sd=("reward","std"),reward_per_day_mean=("reward_per_day","mean"),length_mean=("length","mean")).reset_index()
        table["group"]=rec["group"];table["seed"]=rec["seed"];curves.append(table)
        replaced=ep.source!="real"
        # Legacy scenario=1 means the middle fixed stress scenario, not a tail
        # sampling pool. Do not mix those unrelated label meanings in summaries.
        tail_all=np.nan if rec["group"] in E.LEGACY else ((ep.scenario==1)&replaced).mean()*100
        tail_replaced=np.nan if rec["group"] in E.LEGACY else float((ep.loc[replaced,"scenario"]==1).mean()*100) if replaced.any() else 0.
        summaries.append({"group":rec["group"],"seed":rec["seed"],"episodes":len(ep),"replacement_pct":replaced.mean()*100,
            "synthetic_pct":(ep.source=="synthetic").mean()*100,"real_replay_pct":(ep.source=="real_replay").mean()*100,
            "tail_pool_all_episodes_pct":tail_all,
            "tail_pool_replacement_pct":tail_replaced,
            "mean_length":ep.length.mean()})
    pd.DataFrame(summaries).to_csv(E.OUT/"episode_sampling.csv",index=False)
    pd.concat(curves,ignore_index=True).to_csv(E.OUT/"training_episode_curves.csv",index=False)


def subperiods(selected):
    rows=[];weights=[]
    for rec in selected.itertuples():
        d=pd.read_csv(ROOT/rec.daily_path,parse_dates=["date"])
        start,end=E.PERIODS[rec.split]
        spans=[(str(y),f"{y}-01-01",f"{y}-12-31") for y in range(int(start[:4]),int(end[:4])+1)]
        if rec.split=="test":spans += [("2020-2022","2020-01-01","2022-12-31"),("2023-2025","2023-01-01","2025-12-31"),("2020Q1","2020-01-01","2020-03-31")]
        for label,a,b in spans:
            lo=d.date.searchsorted(pd.Timestamp(a));hi=d.date.searchsorted(pd.Timestamp(b),side="right")
            assert lo>=1 and hi>lo
            z=d.iloc[lo-1:hi].copy()
            if a!=start:
                z["value"]=z.value/z.value.iloc[0]
                z.loc[z.index[0],"trf_mu"]=1.
            metric={**E.metrics(z),**horizon_metrics(z.value.to_numpy(float))}
            rows.append({"group":rec.group,"seed":rec.seed,"split":rec.split,"period":label,**metric})
        for t in ["CASH",*E.tickers()]:
            weights.append({"group":rec.group,"seed":rec.seed,"split":rec.split,"asset":t,"mean_weight_pct":d[f"weight_{t}"].mean()*100,"max_weight_pct":d[f"weight_{t}"].max()*100})
    f=pd.DataFrame(rows);f.to_csv(E.OUT/"selected_subperiod_metrics.csv",index=False)
    f.groupby(["group","split","period"])[METRICS].agg(["mean","std","median"]).to_csv(E.OUT/"subperiod_summary.csv")
    pd.DataFrame(weights).to_csv(E.OUT/"selected_asset_weights.csv",index=False)


def paired_differences(selected):
    comparisons=[("FULL_B50","REAL_ONLY"),("REAL_REPLAY_PR30_B50","REAL_ONLY"),("REAL_REPLAY_PR10_B50","REAL_ONLY"),("PR30_B50","REAL_ONLY"),("PR10_B50","REAL_ONLY"),
                 ("PR30_B50","FULL_B50"),("PR10_B50","FULL_B50"),("PR30_B50","REAL_REPLAY_PR30_B50"),("PR10_B50","REAL_REPLAY_PR10_B50"),("PR10_B50","PR30_B50")]
    rng=np.random.default_rng(20261019);indices=rng.integers(0,10,size=(100000,10));rows=[]
    for split in E.PERIODS:
        for a,b in comparisons:
            x=selected[(selected.group==a)&(selected.split==split)].set_index("seed").sort_index()
            y=selected[(selected.group==b)&(selected.split==split)].set_index("seed").sort_index()
            assert x.index.tolist()==y.index.tolist()==list(range(10))
            for metric in ["return_pct","cagr_net_pct","sharpe","calmar","mdd_pct","es95_daily_pct","es95_21d_pct","worst_21d_return_pct","mean_largest_stock_weight_pct"]:
                delta=(x[metric]-y[metric]).to_numpy();ci=np.quantile(delta[indices].mean(axis=1),[.025,.975])
                negative=metric in ["mdd_pct","es95_daily_pct","es95_21d_pct","mean_largest_stock_weight_pct"]
                rows.append({"split":split,"treatment":a,"control":b,"metric":metric,"mean_difference":delta.mean(),"ci95_low":ci[0],"ci95_high":ci[1],"better_seeds":int((delta<0 if negative else delta>0).sum()),"n_pairs":10})
    pd.DataFrame(rows).to_csv(E.OUT/"paired_seed_differences.csv",index=False)


def configuration_table():
    rows=[]
    for group in E.GROUPS:
        rec=E.records([group])[0];cfg=json.loads((Path(rec["run"])/"config.json").read_text())
        fee=cfg.get("train_commission")
        if fee is None:fee=cfg["commission"]
        with zipfile.ZipFile(Path(rec["run"])/"ckpt/step_025000.zip") as z:
            system=z.read("system_info.txt").decode()
            checkpoint_config=json.loads(z.read("data"))
        common={"learning_rate":.0003,"n_steps":2048,"batch_size":256,"n_epochs":10,"gamma":.99,"gae_lambda":.95,"ent_coef":0.}
        for key,value in common.items():assert checkpoint_config[key]==value,(group,key,checkpoint_config[key])
        info=dict(line[2:].split(": ",1) for line in system.splitlines() if line.startswith("- ") and ": " in line)
        rows.append({"group":group,"n_seeds":3 if group in E.LEGACY else 10,"training_trf":"legacy" if group in E.LEGACY else "fixed",
                     "train_commission":fee,"evaluation_commission":.0025,"replacement_probability":cfg["p"],
                     "replacement_source":"paired_real" if group.startswith("REAL_REPLAY") else "generated" if cfg["p"] else "none",
                     "condition_policy":"legacy_fixed_scenarios" if group in E.LEGACY and cfg["p"] else "full100" if group=="FULL_B50" else "full70_tail30_PR30" if "PR30" in group else "full70_tail30_PR10" if "PR10" in group else "ordinary_real",
                     **common,"torch_version":info.get("PyTorch"),"sb3_version":info.get("Stable-Baselines3"),"training_os":info.get("OS"),"source_run":str(Path(rec["run"]).relative_to(ROOT))})
    pd.DataFrame(rows).to_csv(E.OUT/"group_configurations.csv",index=False)


def legacy_model_equivalence():
    import torch
    import io
    rows=[]
    for rec in E.records([g for g in E.LEGACY if g.startswith("V2_")]):
        group=rec["group"].removeprefix("V2_");seed=rec["seed"]
        old=ROOT/"results/scenario_mix/v2_sharpe"/f"{group}_seed{seed}"/f"{group}_seed{seed}.zip"
        val=pd.read_csv(Path(rec["run"])/"valid.csv").sort_values("timesteps")
        step=int(val.loc[val.valid_sharpe.idxmax(),"timesteps"])
        new=Path(rec["run"])/"ckpt"/f"step_{step:06d}.zip"
        states=[]
        for p in [old,new]:
            with zipfile.ZipFile(p) as z: states.append(torch.load(io.BytesIO(z.read("policy.pth")),map_location="cpu",weights_only=True))
        equal=set(states[0])==set(states[1]) and all(torch.equal(states[0][k],states[1][k]) for k in states[0])
        assert equal,(rec,step)
        rows.append({"group":rec["group"],"seed":seed,"original_sharpe_selected_step":step,"policy_equal":equal,"saved_v2_model_sha256":E.sha(old),"logged_checkpoint_sha256":E.sha(new)})
    pd.DataFrame(rows).to_csv(E.OUT/"v2_saved_model_equivalence.csv",index=False)


def collect():
    freeze=json.loads((E.OUT/"frozen_selection.json").read_text())
    assert freeze["prices_sha256"]==E.sha(E.PRICES)
    training=audit_training();frames=[];provenance=[]
    for rec in E.records():
        p=E.OUT/"evaluations"/rec["id"]
        for phase in ["pretest","test"]:
            metadata=json.loads((p/f"{phase}_complete.json").read_text())
            assert metadata["inputs"]["evaluator_sha256"]==E.sha(Path(E.__file__))
            for name,digest in metadata["artifacts"].items():assert E.sha(p/name)==digest,p/name
            frames.append(pd.read_csv(p/f"{phase}_metrics.csv"))
            provenance.append({"path":str((p/f"{phase}_complete.json").relative_to(ROOT)),"sha256":E.sha(p/f"{phase}_complete.json")})
    all_rows=pd.concat(frames,ignore_index=True)
    keys=["group","seed","nominal_timesteps","split"]
    assert len(all_rows)==5400 and not all_rows.duplicated(keys).any()
    assert set(all_rows.group)==set(E.GROUPS)
    horizon=[]
    for row in all_rows.itertuples():
        values=pd.read_csv(ROOT/row.daily_path,usecols=["value"]).value.to_numpy(float)
        horizon.append(horizon_metrics(values))
    all_rows=pd.concat([all_rows.reset_index(drop=True),pd.DataFrame(horizon)],axis=1)
    assert np.isfinite(all_rows[METRICS]).all().all()
    choices=pd.DataFrame(freeze["selected"])
    selected=all_rows.merge(choices[["group","seed","nominal_timesteps"]],on=["group","seed","nominal_timesteps"],validate="many_to_one")
    assert len(selected)==270 and selected.groupby(["group","seed"]).size().eq(3).all()
    for row in selected.itertuples():
        d=pd.read_csv(ROOT/row.daily_path)
        # Independently verify the two principal metrics from stored daily values.
        returns=np.log(d.value).diff().dropna()
        sharpe=float(returns.mean()/returns.std(ddof=1)*np.sqrt(252))
        np.testing.assert_allclose([sharpe,(d.value.iloc[-1]-1)*100],[row.sharpe,row.return_pct],rtol=1e-9,atol=1e-9)
    all_rows.to_csv(E.OUT/"all_checkpoint_metrics.csv",index=False)
    selected.to_csv(E.OUT/"selected_metrics.csv",index=False)
    group_summary(selected).to_csv(E.OUT/"selected_summary.csv",index=False)
    group_summary(selected[selected.seed<3]).to_csv(E.OUT/"selected_summary_common_seeds0_2.csv",index=False)
    last=all_rows[all_rows.nominal_timesteps==500000]
    group_summary(last).to_csv(E.OUT/"fixed_final_summary.csv",index=False)
    sens=selected[["group","seed","split","nominal_timesteps",*METRICS]].merge(last[["group","seed","split",*METRICS]],on=["group","seed","split"],suffixes=("_selected","_final"),validate="one_to_one")
    sens.to_csv(E.OUT/"checkpoint_selection_sensitivity.csv",index=False)
    old_choices=choices[["group","seed","original_log_selected_step"]].rename(columns={"original_log_selected_step":"nominal_timesteps"})
    historical=all_rows.merge(old_choices,on=["group","seed","nominal_timesteps"],validate="many_to_one")
    historical.to_csv(E.OUT/"original_selection_common_evaluation.csv",index=False)
    benchmarks();episode_tables();subperiods(selected);paired_differences(selected);configuration_table();legacy_model_equivalence()
    E.write(E.OUT/"analysis_manifest.json",{"status":"tables_complete","generated_at":datetime.now(timezone.utc).isoformat(),"groups":E.GROUPS,"runs":90,"checkpoints":1800,"split_evaluations":5400,"selected_split_rows":270,
         "training":training,"frozen_selection_sha256":E.sha(E.OUT/"frozen_selection.json"),"evaluation_manifests":provenance,
         "definitions":{"train":"Deterministic continuous real 2010-2017 evaluation; distinct from sampled stochastic episode rewards.","return_pct":"Cumulative net return including entry cost, relative to pre-entry capital=1.","cagr_net_pct":"final_value ** (252 / trading_days) - 1, including entry cost.","annual_return_pct":"exp(mean daily log returns *252)-1; excludes the initial entry-cost level. Used in legacy-compatible Calmar.","sharpe":"Mean daily log return / sample SD * sqrt(252), rf=0.","sortino":"Mean daily log return / sqrt(mean(min(log return,0)^2)) * sqrt(252).","mdd_pct":"Positive magnitude from daily NAV running maximum.","es95_daily_pct":"Negative mean of simple daily returns <= empirical fifth percentile; losses positive, gains negative.","horizon21":"Overlapping 21-trading-day simple NAV returns within each split. Worst return keeps the return sign; VaR/ES report losses as positive and gains as negative. Overlapping windows are not independent observations.","fee_factor_drag_pct":"1-product(TRF multipliers including entry); conditional on fixed actions, not a separate optimized no-cost strategy.","subperiods":"Carry existing holdings and policy state. Later subperiod NAV is rebased to the preceding day's NAV; only a split's first subperiod includes entry cost.","bootstrap":"100000 paired-seed resamples, percentile 95% interval; unadjusted exploratory comparisons, training-seed uncertainty only.","bands":"Across-seed IQR unless stated otherwise; not a confidence interval."},
         "limitations":["Validation is used for selection and is therefore optimistic.","All checkpoint test curves are post-selection diagnostics; no test maximum is used to select models.","The old v2/v3 test periods have been inspected previously; these are not a pristine blind holdout.","Legacy v2/v3 have three seeds, legacy TRF training, different fixed-condition generation and runtime; v3 training has zero fees. Common reevaluation does not remove training confounding.","Current generated datasets share one TimeDiff checkpoint and one sampling/noise seed; results do not measure generator-seed uncertainty.","Real replay duplicates preserve matching sampling weights and do not represent extra independent market events.","Generator conditions use original prices; current PPO uses the extended price download, with measured <=1.282 ppm overlap price differences."]})
    print("Tables complete: 16 groups, 90 runs, 1800 checkpoints, 5400 split evaluations",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["collect","audit-training"]);args=parser.parse_args()
    collect() if args.command=="collect" else audit_training()
