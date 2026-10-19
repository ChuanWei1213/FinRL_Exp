"""Final structural, numerical, selection-timing and artifact-integrity checks."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from analysis.scenario_mix.comprehensive_20261019 import evaluate as E
import numpy as np
import pandas as pd
from PIL import Image


def main():
    all_rows=pd.read_csv(E.OUT/"all_checkpoint_metrics.csv")
    selected=pd.read_csv(E.OUT/"selected_metrics.csv")
    freeze=json.loads((E.OUT/"frozen_selection.json").read_text())
    frozen_at=datetime.fromisoformat(freeze["frozen_at"])
    assert len(all_rows)==5400 and len(selected)==270 and len(freeze["selected"])==90
    assert set(all_rows.group)==set(E.GROUPS)
    assert all_rows.groupby(["group","seed","split"]).size().eq(20).all()
    assert not all_rows.duplicated(["group","seed","split","nominal_timesteps"]).any()
    assert selected.groupby(["group","seed"]).nominal_timesteps.nunique().eq(1).all()
    assert E.sha(E.PRICES)==freeze["prices_sha256"]
    status_by_phase={"pretest":{},"test":{}}
    sources=[]
    test_starts=[]
    for rec in E.records():
        for phase in ["pretest","test"]:
            p=E.OUT/"evaluations"/rec["id"]/f"{phase}_complete.json"
            d=json.loads(p.read_text())
            assert d["inputs"]["prices_sha256"]==freeze["prices_sha256"]
            assert d["inputs"]["evaluator_sha256"]==E.sha(E.__file__)
            for item in d["inputs"]["checkpoints"]:
                assert E.sha(ROOT/item["checkpoint"])==item["checkpoint_sha256"]
            if phase=="test":
                inferred_start=datetime.fromisoformat(d["completed_at"])-timedelta(seconds=d["elapsed_seconds"])
                assert inferred_start>frozen_at,(rec,inferred_start,frozen_at)
                test_starts.append(inferred_start)
            status_by_phase[phase][rec["id"]]={"state":"succeeded","rows":d["rows"],"completed_at":d["completed_at"],"elapsed_seconds":d["elapsed_seconds"]}
            sources.append({"path":str(p.relative_to(ROOT)),"sha256":E.sha(p)})
    for phase,status in status_by_phase.items():E.write(E.OUT/f"evaluation_{phase}_status.json",status)
    max_sum_error=0.;max_metric_error=0.
    for row in selected.itertuples():
        d=pd.read_csv(ROOT/row.daily_path,parse_dates=["date"])
        weights=d[[f"weight_{a}" for a in ["CASH",*E.tickers()]]].to_numpy()
        assert np.isfinite(weights).all() and weights.min()>-1e-7 and weights.max()<1+1e-7
        max_sum_error=max(max_sum_error,float(np.abs(weights.sum(axis=1)-1).max()))
        assert max_sum_error<1e-6
        change=d.value.pct_change().iloc[1:]
        sharpe=np.log1p(change).mean()/np.log1p(change).std(ddof=1)*np.sqrt(252)
        mdd=100*(1-d.value/d.value.cummax()).max()
        max_metric_error=max(max_metric_error,float(abs(sharpe-row.sharpe)),float(abs(mdd-row.mdd_pct)))
        assert np.isclose(sharpe,row.sharpe,atol=1e-9,rtol=1e-9)
        assert np.isclose(mdd,row.mdd_pct,atol=1e-9,rtol=1e-9)
    figures=[]
    for p in sorted((E.OUT/"figures").rglob("*.png")):
        with Image.open(p) as im:
            assert im.width>=1000 and im.height>=500
            size=list(im.size);im.verify()
        svg=p.with_suffix(".svg");assert svg.exists() and "<svg" in svg.read_text()
        figures.append({"path":str(p.relative_to(ROOT)),"size":size,"sha256":E.sha(p),"svg_sha256":E.sha(svg)})
    assert len(figures)==78,len(figures)
    visual_path=E.OUT/"visual_review.json"
    visual=json.loads(visual_path.read_text())
    assert visual["status"]=="passed"
    assert visual["figures"]=={item["path"]:item["sha256"] for item in figures}
    findings=E.OUT/"findings.json"
    assert json.loads(findings.read_text())["paragraphs"]
    notebook=Path(__file__).with_name("comparison.ipynb")
    nb=json.loads(notebook.read_text());code=[c for c in nb["cells"] if c["cell_type"]=="code"]
    assert len(code)==10 and all(c["execution_count"] is not None for c in code)
    assert not any(o["output_type"]=="error" for c in code for o in c["outputs"])
    notebook_audit=json.loads((E.OUT/"notebook_execution.json").read_text())
    assert notebook_audit["errors"]==0 and notebook_audit["embedded_images"]==18
    training=json.loads((E.NEW/"completion_summary.json").read_text())
    assert datetime.fromisoformat(training["completed_at"])<frozen_at<min(test_starts)
    source_files={str(p.relative_to(ROOT)):E.sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    table_files={p.name:E.sha(p) for p in sorted(E.OUT.glob("*.csv"))}
    result={"status":"verified","verified_at":datetime.now(timezone.utc).isoformat(),"groups":16,"runs":90,"new_training_runs":30,"checkpoints":1800,"split_evaluations":5400,"selected_split_rows":270,
            "training_completed_at":training["completed_at"],"selection_frozen_at":freeze["frozen_at"],"earliest_test_start_inferred":min(test_starts).isoformat(),
            "maximum_selected_weight_sum_error":max_sum_error,"maximum_independent_metric_error":max_metric_error,
            "all_original_checkpoints_match_evaluation_hashes":True,"figures":figures,"evaluation_manifests":sources,
            "source_sha256":source_files,"table_sha256":table_files,"notebook":{"path":str(notebook.relative_to(ROOT)),"sha256":E.sha(notebook),**notebook_audit},
            "findings_sha256":E.sha(findings),
            "visual_review":{"path":str(visual_path.relative_to(ROOT)),"sha256":E.sha(visual_path),"figure_count":len(visual["figures"]),"note":"Recorded separately after visual inspection; structural checks alone are insufficient."}}
    E.write(E.OUT/"completion_audit.json",result)
    print(json.dumps({k:result[k] for k in ["status","groups","runs","new_training_runs","checkpoints","split_evaluations","maximum_independent_metric_error"]},indent=2))


if __name__=="__main__":main()
