"""Read-only generation comparison and input audit for the new control study."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.scenario_mix.comprehensive_20261019 import evaluate as E
import numpy as np
import pandas as pd


def main():
    base=ROOT.parent/"TimeDiff/outputs/dow10/conditional/empirical_joint/run_20261017/batches"
    population=pd.read_csv(base/"full70_tail30_pr30/population.csv")
    cuts={q:float(np.quantile(population.target_return,q/100)) for q in [10,30]}
    rows=[];audits=[];hashes=[]
    for label,folder in [("Full100","full100_20261019"),("Full70+PR30","full70_tail30_pr30"),("Full70+PR10","full70_tail30_pr10")]:
        path=base/folder;meta=json.loads((path/"generation.json").read_text())
        assert meta["status"]=="complete" and meta["n_generated_paths"]==2000
        hashes.append(meta["checkpoint_sha256"])
        z=dict(np.load(path/"paths.npz",allow_pickle=False));n=len(z["close"])
        assert n==2000 and z["close"].shape==(2000,21,10)
        assert np.isfinite(z["raw_features"]).all()
        for key in ["close","high","low"]:assert np.isfinite(z[key]).all() and (z[key]>0).all()
        assert (z["low"]<=z["close"]).all() and (z["close"]<=z["high"]).all()
        previous=np.concatenate([z["anchor_close"][:,None,:],z["close"][:,:-1,:]],axis=1)
        stock_returns=np.log(z["close"].astype(float)/previous)
        rp=np.log(np.exp(stock_returns).mean(axis=2))
        realized=np.column_stack([np.expm1(rp.sum(axis=1)),rp.std(axis=1,ddof=1)*np.sqrt(252)])
        np.testing.assert_allclose(realized,z["realized"],atol=3e-7,rtol=0)
        unique=len(np.unique(z["raw_features"][...,::3].reshape(n,-1),axis=0))
        corrs=[]
        for r in stock_returns:
            c=np.corrcoef(r.T);corrs.append(c[np.triu_indices(10,1)].mean())
        real_corrs=np.array([np.corrcoef(r.T)[np.triu_indices(10,1)].mean() for r in z["paired_real_future"][...,::3]])
        row={"dataset":label,"paths":n,"unique_histories":len(np.unique(z["window_id"])),"unique_close_paths":unique,
             "target_return_mean_pct":z["target"][:,0].mean()*100,"realized_return_mean_pct":z["realized"][:,0].mean()*100,
             "target_vol_mean_pct":z["target"][:,1].mean()*100,"realized_vol_mean_pct":z["realized"][:,1].mean()*100,
             "return_mae_pp":abs(z["realized"][:,0]-z["target"][:,0]).mean()*100,
             "vol_mae_pp":abs(z["realized"][:,1]-z["target"][:,1]).mean()*100,
             "mean_stock_correlation":float(np.mean(corrs)),"paired_real_stock_correlation":float(real_corrs[z["draw_id"]].mean()),"high_correction_pct":(z["raw_features"][...,1::3]<0).mean()*100,
             "low_correction_pct":(z["raw_features"][...,2::3]>0).mean()*100,
             "extra_tail_pool_pct":(z["scenario"]==1).mean()*100}
        for q,cut in cuts.items():
            row[f"target_pr{q}_pct"]=(z["target"][:,0]<=cut).mean()*100
            row[f"realized_pr{q}_pct"]=(z["realized"][:,0]<=cut).mean()*100
        rows.append(row)
        if label=="Full100":assert (z["scenario"]==0).all()
        audits.append({"dataset":label,"path":str(path/"paths.npz"),"sha256":E.sha(path/"paths.npz"),"recomputed_condition_max_abs_error":float(abs(realized-z["realized"]).max()),"unique_close_paths":unique})
    assert len(set(hashes))==1
    pd.DataFrame(rows).to_csv(E.OUT/"generation_quality_comparison.csv",index=False)
    E.write(E.OUT/"generation_quality_audit.json",{"same_generator_checkpoint_sha256":hashes[0],"tail_cutoffs":cuts,"datasets":audits,"note":"No paths removed based on quality or realized return; all 2000 retained per dataset."})
    legacy=ROOT.parent/"TimeDiff/outputs/dow10/conditional/fixed_scenarios/legacy/rl_paths/paths.npz"
    old=dict(np.load(legacy,allow_pickle=False))
    targets,counts=np.unique(old["target"],axis=0,return_counts=True)
    E.write(E.OUT/"legacy_synthetic_profile.json",{"source":str(legacy),"sha256":E.sha(legacy),"paths":len(old["close"]),
        "unique_anchor_dates":len(np.unique(old["anchor_dates"])),"anchor_min":min(map(str,old["anchor_dates"])),"anchor_max":max(map(str,old["anchor_dates"])),
        "target_return_volatility_pairs":targets.tolist(),"counts":counts.tolist(),"realized_return_mean":float(old["realized"][:,0].mean()),
        "realized_volatility_mean":float(old["realized"][:,1].mean()),"interpretation":"These actual NPZ targets override script defaults when describing the historical experiment."})
    full=np.load(base/"full100_20261019/paths.npz",allow_pickle=False)
    shared=[]
    for folder in ["full70_tail30_pr30","full70_tail30_pr10"]:
        mixed=np.load(base/folder/"paths.npz",allow_pickle=False);mask=mixed["scenario"]==0
        assert mask.sum()==1400
        for key in ["window_id","target","raw_features","close","high","low"]:
            np.testing.assert_array_equal(full[key][mask],mixed[key][mask])
        shared.append({"paired_dataset":folder,"shared_full_pool_paths":1400,"all_features_and_ohlc_bitwise_equal":True,"changed_component_paths":600})
    E.write(E.OUT/"shared_full_component_audit.json",{"comparisons":shared,"note":"Same plan seed and diffusion stream preserve all 1400 full-component paths exactly; only the other 600 paths' sampling conditions change."})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
