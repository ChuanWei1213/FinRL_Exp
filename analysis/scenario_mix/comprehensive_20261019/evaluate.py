"""Reevaluate all checkpoints on a common price source, fee formula and splits.

Pretest: train/validation only, then freeze validation-Sharpe selections.
Test: requires all training jobs succeeded and a frozen selection manifest.
Existing experiment files are read only; per-run results are resumable by hashes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/scenario_mix/comprehensive_20261019"
NEW = ROOT / "results/scenario_mix/joint_missing_controls_seeds0_9_20261019_run01"
MIXED = ROOT / "results/scenario_mix/joint_pr30_pr10_b50_seeds0_9_20261018_run02"
REAL = ROOT / "results/scenario_mix/joint_real_only_seeds0_9_20261018_run01"
PRICES = ROOT.parent / "TimeDiff/datasets/dow10/prices_2004_2025.csv"
PERIODS = {"train": ("2010-01-01", "2017-12-31"), "val": ("2018-01-01", "2019-12-31"), "test": ("2020-01-01", "2025-12-31")}
CURRENT = ["REAL_ONLY", "REAL_REPLAY_PR30_B50", "REAL_REPLAY_PR10_B50", "FULL_B50", "PR30_B50", "PR10_B50"]
LEGACY = [f"{v}_{g}" for v in ["V2", "V3"] for g in ["A", "B20", "B50", "B70", "B90"]]
GROUPS = CURRENT + LEGACY
STEPS = list(range(25000, 500001, 25000))

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-comprehensive-20261019/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/finrl-comprehensive-20261019/cache")
sys.modules.setdefault("quantstats", None)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    tmp.replace(p)


def records(groups=None):
    rows = []
    for group in (groups or GROUPS):
        seeds = range(3) if group in LEGACY else range(10)
        for seed in seeds:
            if group in LEGACY:
                version, old = group.split("_", 1)
                root = ROOT / "results/scenario_mix" / ("v2_rewardlog" if version == "V2" else "v3_nocost")
                run = root / f"{old}_seed{seed}"
            else:
                root = REAL if group == "REAL_ONLY" else MIXED if group in ["PR30_B50", "PR10_B50"] else NEW
                run = root / f"{group}_seed{seed}"
            rows.append({"group": group, "seed": seed, "run": str(run), "id": f"{group}_seed{seed}"})
    return rows


def metrics(d):
    v = d.value.to_numpy(float)
    lr = np.diff(np.log(v)); r = np.expm1(lr)
    dd = v / np.maximum.accumulate(v) - 1
    ann = np.expm1(lr.mean()*252)
    sd = lr.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(lr, 0)**2))
    q = np.quantile(r, .05)
    result = {
        "days": len(lr), "final_value": v[-1], "return_pct": (v[-1]-1)*100,
        "cagr_net_pct": (v[-1]**(252/len(lr))-1)*100,
        "annual_return_pct": ann*100, "volatility_pct": sd*np.sqrt(252)*100,
        "sharpe": lr.mean()/sd*np.sqrt(252), "sortino": lr.mean()/downside*np.sqrt(252),
        "calmar": ann/abs(dd.min()), "mdd_pct": -dd.min()*100,
        "var95_daily_pct": -q*100, "es95_daily_pct": -r[r<=q].mean()*100,
        "positive_day_pct": (r>0).mean()*100, "worst_day_pct": r.min()*100,
        "stock_weight_pct": (1-d.cash_weight.mean())*100,
        "annual_turnover": d.turnover.iloc[1:].mean()*252,
        "fee_factor_drag_pct": (1-d.trf_mu.prod())*100,
        "mean_largest_stock_weight_pct": d[[f"weight_{t}" for t in tickers()]].max(axis=1).mean()*100,
        "max_single_stock_weight_pct": d[[f"weight_{t}" for t in tickers()]].to_numpy().max()*100,
        "concentrated_over80_pct": (d[[f"weight_{t}" for t in tickers()]].max(axis=1)>.8).mean()*100,
        "action_bound_fraction_pct": d.action_bound_fraction.iloc[1:].mean()*100,
    }
    assert all(np.isfinite(x) for x in result.values()), result
    return {k: float(x) for k, x in result.items()}


def tickers():
    from finrl.experiments.scenario_mix_common import TICKERS
    return TICKERS


def rollout(model, market, start, end, commission=.0025):
    from finrl.experiments import scenario_mix as S
    s,e = market.idx(start),market.last_idx(end)
    if s<50 or e<s or market.dates[e] < pd.Timestamp(end)-pd.Timedelta(days=7):
        raise ValueError(f"Incomplete requested evaluation period {start}..{end}")
    sl=slice(s-50,e+1)
    frame=S.episode_frame(market.dates[sl], market.close[sl],market.high[sl],market.low[sl],market.tickers)
    inner=S._InnerEpisode(frame,len(market.tickers),commission,50)
    obs,_=inner.reset(); obs=S.normalize_obs(obs)
    mus,bounds=[],[]
    done=False
    while not done:
        action=model.predict(obs,deterministic=True)[0]
        bounds.append(float(np.isclose(np.abs(action),1.,atol=1e-6).mean()))
        obs,_,done,_,info=inner.step(action)
        mus.append(float(info["trf_mu"]))
        obs=S.normalize_obs(obs)
    env=inner.env; n=len(market.tickers)
    entry=1-commission*n/(n+1)
    final=np.asarray(env._final_weights,float); acts=np.asarray(env._actions_memory,float)
    d=pd.DataFrame({"date":market.dates[s-1:e+1],"value":np.asarray(env._asset_memory["final"],float)/env._initial_amount*entry,
                    "cash_weight":final[:,0],"turnover":np.r_[n/(n+1),np.abs(acts[1:]-final[:-1]).sum(axis=1)/2],
                    "trf_mu":np.r_[entry,mus],"action_bound_fraction":np.r_[np.nan,bounds]})
    for i,t in enumerate(["CASH",*market.tickers]):
        d[f"weight_{t}"]=final[:,i];d[f"target_{t}"]=acts[:,i]
    assert len(d)==e-s+2 and np.isfinite(d.value).all() and (d.value>0).all()
    np.testing.assert_allclose(final.sum(axis=1),1,atol=1e-6,rtol=0)
    return d


def checkpoint_inventory(record):
    run=Path(record["run"])
    cfg=json.loads((run/"config.json").read_text())
    v=pd.read_csv(run/"valid.csv")
    assert v.timesteps.tolist()==STEPS and len(list((run/"ckpt").glob("*.zip")))==20,record
    for key,value in [("train",list(PERIODS["train"])),("valid",list(PERIODS["val"])),("timesteps",500000),("eval_freq",25000),("commission",.0025),("k_max",21),("log_std_init",-2.),("ent_coef",0.)]:
        assert cfg[key]==value,(record,key,cfg[key])
    result=[]
    for step in STEPS:
        p=run/"ckpt"/f"step_{step:06d}.zip"
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            meta=json.loads(z.read("data"))
        assert meta["seed"]==record["seed"] and meta["num_timesteps"]==step//25000*26624
        result.append({"nominal_timesteps":step,"actual_timesteps":meta["num_timesteps"],"checkpoint":str(p.relative_to(ROOT)),"checkpoint_sha256":sha(p)})
    return cfg,v,result


def evaluate_run(record,phase):
    import torch
    from stable_baselines3 import PPO
    from finrl.experiments import scenario_mix as S
    torch.set_num_threads(1)
    started=time.monotonic()
    cfg,original,inventory=checkpoint_inventory(record)
    destination=OUT/"evaluations"/record["id"]
    destination.mkdir(parents=True,exist_ok=True)
    inputs={"prices_sha256":sha(PRICES),"checkpoints":inventory,"config_sha256":sha(Path(record["run"])/"config.json"),"evaluator_sha256":sha(__file__)}
    marker=destination/f"{phase}_complete.json"
    if marker.exists():
        saved=json.loads(marker.read_text())
        assert saved["inputs"]==inputs,("Cached input changed",marker)
        for rel,digest in saved["artifacts"].items(): assert sha(destination/rel)==digest
        return {"id":record["id"],"cached":True}
    if phase=="test":
        selection_path=OUT/"frozen_selection.json"
        freeze=json.loads(selection_path.read_text())
        assert freeze["prices_sha256"]==inputs["prices_sha256"]
        assert len(freeze["selected"])==90
        assert all(s["state"]=="succeeded" for s in json.loads((NEW/"status.json").read_text()).values())
    market=S.MarketData(PRICES,tickers())
    windows=["train","val"] if phase=="pretest" else ["test"]
    rows=[]; artifacts={}
    for item in inventory:
        model=PPO.load(ROOT/item["checkpoint"],device="cpu")
        assert model.num_timesteps==item["actual_timesteps"]
        for split in windows:
            d=rollout(model,market,*PERIODS[split]);m=metrics(d)
            row={"group":record["group"],"seed":record["seed"],"split":split,**item,**m}
            if split=="val":
                saved=original[original.timesteps==item["nominal_timesteps"]].iloc[0]
                row["original_valid_sharpe"]=float(saved.valid_sharpe)
                row["original_valid_final_value"]=float(saved.valid_final_value)
                if record["group"] in CURRENT:
                    np.testing.assert_allclose([m["sharpe"],m["final_value"]],[saved.valid_sharpe,saved.valid_final_value],rtol=1e-7,atol=1e-7)
            name=f"step_{item['nominal_timesteps']:06d}_{split}.csv.gz"
            d.to_csv(destination/name,index=False,compression={"method":"gzip","mtime":0})
            artifacts[name]=sha(destination/name)
            row["daily_path"]=str((destination/name).relative_to(ROOT))
            rows.append(row)
    metrics_name=f"{phase}_metrics.csv"
    pd.DataFrame(rows).to_csv(destination/metrics_name,index=False)
    artifacts[metrics_name]=sha(destination/metrics_name)
    write(marker,{"completed_at":datetime.now(timezone.utc).isoformat(),"inputs":inputs,"artifacts":artifacts,"rows":len(rows),"elapsed_seconds":time.monotonic()-started})
    return {"id":record["id"],"phase":phase,"elapsed_seconds":time.monotonic()-started,"rows":len(rows)}


def freeze():
    target=OUT/"frozen_selection.json"
    if target.exists():
        raise FileExistsError("Selections are already frozen")
    selected=[];manifests={}
    for rec in records():
        p=OUT/"evaluations"/rec["id"]
        marker=json.loads((p/"pretest_complete.json").read_text())
        frame=pd.read_csv(p/"pretest_metrics.csv")
        assert len(frame)==40
        v=frame[frame.split=="val"].sort_values("nominal_timesteps")
        best=v.loc[v.sharpe.idxmax()].to_dict()
        old=v.loc[v.original_valid_sharpe.idxmax()]
        selected.append({"group":rec["group"],"seed":rec["seed"],"nominal_timesteps":int(best["nominal_timesteps"]),
                         "actual_timesteps":int(best["actual_timesteps"]),"checkpoint":best["checkpoint"],"checkpoint_sha256":best["checkpoint_sha256"],
                         "validation_sharpe":best["sharpe"],"original_log_selected_step":int(old.nominal_timesteps),
                         "selection_changed_from_original_log":int(old.nominal_timesteps)!=int(best["nominal_timesteps"])})
        manifests[str(p.relative_to(ROOT)/"pretest_complete.json")]=sha(p/"pretest_complete.json")
    assert len(selected)==90
    write(target,{"frozen_at":datetime.now(timezone.utc).isoformat(),"rule":"validation Sharpe maximum; earliest tie; corrected TRF and common prices; no test used",
                  "prices_sha256":sha(PRICES),"protocol_sha256":sha(OUT/"protocol.json"),"pretest_manifest_sha256":manifests,"selected":selected})
    pd.DataFrame(selected).to_csv(OUT/"selected_checkpoints.csv",index=False)
    print(f"Frozen {len(selected)} selections before test",flush=True)


def run_phase(phase,workers,groups):
    pending=records(groups)
    if phase=="test" and not (OUT/"frozen_selection.json").exists():
        raise RuntimeError("Freeze validation selections before test")
    state={}
    path=OUT/f"evaluation_{phase}_status.json"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(evaluate_run,rec,phase):rec for rec in pending}
        for f in as_completed(futures):
            rec=futures[f]
            try: result=f.result()
            except Exception as error:
                state[rec["id"]]={"state":"failed","error":repr(error)};write(path,state);raise
            state[rec["id"]]={"state":"succeeded",**result};write(path,state)
            print(json.dumps(result),flush=True)


def benchmark():
    import torch
    from stable_baselines3 import PPO
    from finrl.experiments import scenario_mix as S
    torch.set_num_threads(1)
    rec=records(["REAL_ONLY"])[0]
    model=PPO.load(Path(rec["run"])/"ckpt/step_025000.zip",device="cpu")
    market=S.MarketData(PRICES,tickers())
    rows=[]
    for split in ["train","val"]:
        started=time.monotonic();d=rollout(model,market,*PERIODS[split]);elapsed=time.monotonic()-started
        expected=S.run_continuous(lambda obs:model.predict(obs,deterministic=True)[0],market,*PERIODS[split],.0025)
        for column in ["value","cash_weight","turnover"]:
            np.testing.assert_allclose(d[column],expected[column],rtol=1e-11,atol=1e-11)
        rows.append({"split":split,"rows":len(d),"seconds":elapsed,"reference_max_value_error":float(np.abs(d.value-expected.value).max()),"metrics":metrics(d)})
    write(OUT/"evaluator_preflight.json",rows)
    print(json.dumps(rows,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=["pretest","test","freeze","benchmark"])
    ap.add_argument("--workers",type=int,default=5)
    ap.add_argument("--groups",nargs="+",choices=GROUPS)
    args=ap.parse_args()
    if args.phase=="freeze":freeze()
    elif args.phase=="benchmark":benchmark()
    else:run_phase(args.phase,args.workers,args.groups)


if __name__=="__main__":main()
