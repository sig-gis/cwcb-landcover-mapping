#!/usr/bin/env python3
"""Focused second-round bottom-up models motivated by first-round failures."""
from __future__ import annotations
import csv,json,math,os
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

from src import pipeline as core

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; SEED=20210901

def joint_design(d):
    n,c=d.shape; a=np.zeros((n*12,c*12))
    for p in range(n):
        for m in range(12): a[p*12+m,np.arange(c)*12+m]=d[p]
    return a

def penalty(c,ridge,smooth,pool):
    rows=[]
    if ridge:
        rows.extend(math.sqrt(ridge)*np.eye(c*12))
    if smooth:
        # Adjacent-month first differences; do not impose Dec-Jan continuity.
        for j in range(c):
            for m in range(11):
                r=np.zeros(c*12); r[j*12+m]=math.sqrt(smooth); r[j*12+m+1]=-math.sqrt(smooth); rows.append(r)
    if pool:
        # Shrink each class toward the unweighted class mean for the same month.
        for j in range(c):
            for m in range(12):
                r=np.zeros(c*12); r[np.arange(c)*12+m]=-math.sqrt(pool)/c; r[j*12+m]+=math.sqrt(pool); rows.append(r)
    return np.asarray(rows) if rows else np.empty((0,c*12))

def solve_joint(d,y,config):
    a=joint_design(d); p=penalty(d.shape[1],*config)
    coef=nnls(np.vstack([a,p]),np.r_[y.ravel(),np.zeros(len(p))],maxiter=10000)[0]
    return coef.reshape(d.shape[1],12)

CONFIGS=[
    (10.0,1.0,0.0),
    (10.0,10.0,0.0),
    (10.0,100.0,0.0),
    (10.0,10.0,0.1),
    (10.0,10.0,1.0),
    (10.0,10.0,10.0),
]

def folds(ids,seed):
    return [x for x in np.array_split(np.random.default_rng(seed).permutation(ids),5) if len(x)]

def choose_joint(d,y,train,seed):
    scores={}
    for cfg in CONFIGS:
        errors=[]
        for val in folds(train,seed):
            tr=np.array([i for i in train if i not in set(val)])
            b=solve_joint(d[tr],y[tr],cfg); errors.extend((d[val]@b-y[val]).ravel()**2)
        scores[cfg]=float(np.mean(errors))
    return min(scores,key=scores.get),scores

def lopo_joint(d,y):
    pred=np.zeros_like(y); coefs=[]; selected=[]
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); cfg,_=choose_joint(d,y,tr,SEED+held); selected.append(cfg)
        b=solve_joint(d[tr],y[tr],cfg); pred[held]=d[held]@b; coefs.append(b)
    cfg,scores=choose_joint(d,y,np.arange(len(y)),SEED); full=solve_joint(d,y,cfg)
    return pred,full,np.asarray(coefs),cfg,scores,selected

def solve_annual(d,a,alpha): return core.ridge_nnls(d,a,alpha)

def choose_annual(d,a,train):
    scores={}
    for alpha in core.ALPHAS:
        e=[]
        for val in train:
            tr=train[train!=val]; e.append((d[val]@solve_annual(d[tr],a[tr],alpha)-a[val])**2)
        scores[alpha]=float(np.mean(e))
    return min(scores,key=scores.get)

def lopo_magnitude_shape(d,y):
    pred=np.zeros_like(y); coefs=[]; shapes=[]; selected=[]; annual=y.sum(1)
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); alpha=choose_annual(d,annual,tr); b=solve_annual(d[tr],annual[tr],alpha)
        # Aggregate training curve preserves total volume and is learned without held-out data.
        shape=y[tr].sum(0); shape=shape/shape.sum(); pred[held]=(d[held]@b)*shape
        coefs.append(b); shapes.append(shape); selected.append(alpha)
    alpha=choose_annual(d,annual,np.arange(len(y))); b=solve_annual(d,annual,alpha); shape=y.sum(0)/y.sum()
    return pred,b[:,None]*shape[None,:],np.asarray(coefs),np.asarray(shapes),alpha,selected

def rows_for_metrics(y,preds): return [core.metrics(k,y,v) for k,v in preds.items()]

def write_csv(path,rows): core.write_csv(path,rows)

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    parcels,objects,fragments,classes,_=core.load_spatial(); y=np.array([p["y"] for p in parcels]); d=core.base_matrix(fragments,len(parcels),len(classes))
    mag,mag_full,mag_fold,shapes,mag_alpha,mag_selected=lopo_magnitude_shape(d,y)
    smooth,smooth_full,smooth_fold,cfg,cfg_scores,cfg_selected=lopo_joint(d,y)
    # Recompute first-round reference models from raw data for a self-contained comparison.
    mean=core.mean_lopo(y); simple,simple_full,_,_,_=core.lopo_simple(d,y)
    preds={"mean_curve":mean,"class_area_nnls":simple,"annual_class_area_common_shape":mag,"seasonally_regularized_class_area":smooth}
    metrics=rows_for_metrics(y,preds); paired=core.paired_comparisons(y,preds)
    write_csv(OUT/"refinement_model_comparison.csv",metrics); write_csv(OUT/"refinement_paired_differences.csv",paired)
    parcel=[]
    fulls={"annual_class_area_common_shape":d@mag_full,"seasonally_regularized_class_area":d@smooth_full}
    for name,p in [("annual_class_area_common_shape",mag),("seasonally_regularized_class_area",smooth)]:
        for i,x in enumerate(parcels):
            r={"parcel_fid":x["fid"],"model_id":name,"validation":"LOPO; tuning inside outer training parcels","obs_annual":float(y[i].sum()),"held_annual":float(p[i].sum()),"cosine":core.metrics(name,y[[i]],p[[i]])["mean_cosine_similarity"]}
            for m in range(12): r[f"observed_{m+1:02d}"]=y[i,m]; r[f"held_{m+1:02d}"]=p[i,m]; r[f"full_{m+1:02d}"]=fulls[name][i,m]
            parcel.append(r)
    write_csv(OUT/"refinement_parcel_predictions.csv",parcel)
    rates=[]
    for name,b in [("annual_class_area_common_shape",mag_full),("seasonally_regularized_class_area",smooth_full)]:
        for j,c in enumerate(classes):
            fold_annual=mag_fold[:,j] if name.startswith("annual") else smooth_fold[:,j,:].sum(1)
            r={"model_id":name,"class_final":c,"annual_unit_kgal_per_100m2":float(b[j].sum()),"fold_annual_p05":float(np.quantile(fold_annual,.05)),"fold_annual_p95":float(np.quantile(fold_annual,.95))}
            for m in range(12): r[f"unit_{m+1:02d}"]=b[j,m]
            rates.append(r)
    write_csv(OUT/"refinement_class_curves.csv",rates)
    # Joint parcel-bootstrap uncertainty for the improved challenger. Every draw
    # refits all class-month rates, then the same draw is applied to all objects
    # and aggregated to parcels; no independent object intervals are summed.
    rng=np.random.default_rng(SEED+77); draws=[]
    for _ in range(300):
        sample=rng.integers(0,len(y),len(y)); draws.append(solve_joint(d[sample],y[sample],cfg))
    draws=np.asarray(draws); oq=np.quantile(draws,[.05,.5,.95],axis=0); pd=np.einsum('pc,bcm->bpm',d,draws); pq=np.quantile(pd,[.05,.5,.95],axis=0); paq=np.quantile(pd.sum(2),[.05,.5,.95],axis=0)
    parcel_spatial=[]
    for i,p in enumerate(parcels):
        r={"fid":p["fid"],"model_id":"seasonally_regularized_class_area","validation":"LOPO held prediction; uncertainty is 300 full-data joint parcel-bootstrap refits","cosine":float(np.dot(y[i],smooth[i])/(np.linalg.norm(y[i])*np.linalg.norm(smooth[i]))),"obs_annual":float(y[i].sum()),"held_annual":float(smooth[i].sum()),"full_annual":float((d@smooth_full)[i].sum()),"annual_p05":paq[0,i],"annual_p95":paq[2,i]}
        for m in range(12): r[core.TARGETS[m]]=y[i,m]; r[f"held_{m+1:02d}"]=smooth[i,m]; r[f"full_{m+1:02d}"]=(d@smooth_full)[i,m]; r[f"p05_{m+1:02d}"]=pq[0,i,m]; r[f"p95_{m+1:02d}"]=pq[2,i,m]
        parcel_spatial.append(r)
    write_csv(OUT/"parcel_predictions_seasonally_regularized.csv",parcel_spatial); core.export_gpkg(OUT/"parcel_predictions_seasonally_regularized.gpkg","parcel_predictions",core.INPUT/"parcels_water_2021.gpkg","parcels",parcel_spatial)
    ci={c:i for i,c in enumerate(classes)}; object_rows=[]
    for o in objects:
        j=ci[o["class"]]; r={"fid":o["fid"],"sp_id":o["sp_id"],"class_final":o["class"],"area_m2":o["area_m2"],"model_id":"seasonally_regularized_class_area","support_flag":"supported" if d[:,j].sum()>0 else "unsupported_class","unit":"kgal_per_100m2_month"}
        for m in range(12): r[f"unit_{m+1:02d}"]=smooth_full[j,m]; r[f"object_{m+1:02d}"]=smooth_full[j,m]*o["area_m2"]/core.AREA_UNIT; r[f"unit_p05_{m+1:02d}"]=oq[0,j,m]; r[f"unit_p95_{m+1:02d}"]=oq[2,j,m]; r[f"object_p05_{m+1:02d}"]=oq[0,j,m]*o["area_m2"]/core.AREA_UNIT; r[f"object_p95_{m+1:02d}"]=oq[2,j,m]*o["area_m2"]/core.AREA_UNIT
        annual_draw=draws[:,j,:].sum(1); r["unit_annual"]=float(smooth_full[j].sum()); r["object_annual"]=r["unit_annual"]*o["area_m2"]/core.AREA_UNIT; r["unit_annual_p05"]=float(np.quantile(annual_draw,.05)); r["unit_annual_p95"]=float(np.quantile(annual_draw,.95)); r["object_annual_p05"]=r["unit_annual_p05"]*o["area_m2"]/core.AREA_UNIT; r["object_annual_p95"]=r["unit_annual_p95"]*o["area_m2"]/core.AREA_UNIT; object_rows.append(r)
    write_csv(OUT/"complete_objects_seasonally_regularized.csv",object_rows); core.export_gpkg(OUT/"complete_objects_seasonally_regularized.gpkg","object_predictions",core.INPUT/"landcover_2021.gpkg","superpixels",object_rows)
    direct=d@smooth_full; fragment=np.zeros_like(direct); direct_mag=d@mag_full; fragment_mag=np.zeros_like(direct_mag); byfid={o["fid"]:ci[o["class"]] for o in objects}
    for f in fragments:
        fragment[f["parcel"]]+=f["area_m2"]/core.AREA_UNIT*smooth_full[byfid[f["object_fid"]]]
        fragment_mag[f["parcel"]]+=f["area_m2"]/core.AREA_UNIT*mag_full[byfid[f["object_fid"]]]
    invariant=float(np.max(np.abs(direct-fragment))); invariant_mag=float(np.max(np.abs(direct_mag-fragment_mag)))
    report={"motivation":{"observed_failure":"simple class model improved magnitude but not cosine shape; class curves are weakly identified","focused_models":["annual class-area magnitude with common training-only shape","joint seasonally smoothed and partially pooled nonnegative class-area curves"]},"metrics":metrics,"paired":paired,"settings":{"annual_full_alpha":mag_alpha,"annual_outer_alpha_counts":dict(Counter(mag_selected)),"seasonal_full_config":{"ridge":cfg[0],"smooth":cfg[1],"pool":cfg[2]},"seasonal_outer_config_counts":{str(k):v for k,v in Counter(cfg_selected).items()},"seasonal_full_inner_scores":{str(k):v for k,v in cfg_scores.items()}},"target":core.TARGETS,"units":"thousand gallons/month","aggregation":"both models predict object unit curves before exact area-weighted parcel aggregation","annual_shape_aggregation_max_abs_kgal":invariant_mag,"seasonal_aggregation_max_abs_kgal":invariant,"decision":"seasonally regularized model is a useful challenger but its small paired gain is not statistically decisive"}
    (OUT/"refinement_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    build_report(metrics,paired,report)
    core.build_manifest()
    print(json.dumps({"metrics":metrics,"settings":report["settings"]},indent=2))

def build_report(metrics,paired,result):
    by={x["model_id"]:x for x in metrics}; s=by["class_area_nnls"]; r=by["seasonally_regularized_class_area"]; a=by["annual_class_area_common_shape"]
    pair=next(x for x in paired if x["model_a"]=="class_area_nnls" and x["model_b"]=="seasonally_regularized_class_area")
    text=f"""# Focused second-round model refinement

## Priorities

Preserve the raw monthly target and exact bottom-up aggregation; address the observed magnitude-versus-shape failure; stabilize class curves with minimal added structure; use identical LOPO splits; and reject complexity whose paired improvement is not credible.

## Models tried

**Annual class-area magnitude with common shape.** Nonnegative class-area coefficients estimate annual magnitude. The held-out annual estimate is distributed using the aggregate monthly shape learned only from outer-training parcels. Every object receives its class annual rate times that common shape.

**Seasonally regularized class-area model.** All 12 class-month rates are fitted jointly. Rates remain nonnegative, adjacent months receive a smoothness penalty, and an optional class-pooling penalty is selected inside each outer fold. The full-data selection chose ridge 10, smoothness 10, and no pooling. This is still `area × class unit curve`, not a parcel-only model.

## Results

| model | monthly MAE | RMSE | mean cosine | annual MAE |
|---|---:|---:|---:|---:|
| original class-area | {s['monthly_mae_kgal']:.3f} | {s['monthly_rmse_kgal']:.3f} | {s['mean_cosine_similarity']:.3f} | {s['annual_mae_kgal']:.3f} |
| annual magnitude/common shape | {a['monthly_mae_kgal']:.3f} | {a['monthly_rmse_kgal']:.3f} | {a['mean_cosine_similarity']:.3f} | {a['annual_mae_kgal']:.3f} |
| seasonally regularized | **{r['monthly_mae_kgal']:.3f}** | **{r['monthly_rmse_kgal']:.3f}** | **{r['mean_cosine_similarity']:.3f}** | **{r['annual_mae_kgal']:.3f}** |

The seasonally regularized model improves all four summaries, but monthly MAE improves by only {s['monthly_mae_kgal']-r['monthly_mae_kgal']:.3f} kgal ({100*(s['monthly_mae_kgal']-r['monthly_mae_kgal'])/s['monthly_mae_kgal']:.1f}%). It wins on {pair['parcels_b_better']} of 38 parcels. The paired bootstrap interval for original-minus-regularized MAE is [{pair['bootstrap_p025']:.3f}, {pair['bootstrap_p975']:.3f}] kgal and includes zero. Therefore this is a plausible, well-targeted improvement, not decisive evidence of superiority.

Outer folds also selected different pooling strengths, while the full fit selected none. That instability says the data support temporal smoothing more clearly than hierarchical class pooling.

## Decision

Package the seasonally regularized model as an improved challenger and co-finalist, with complete-object and parcel GeoPackages plus 300 joint parcel-bootstrap refits. Retain the original class-area model as the default when maximum simplicity is preferred. The annual common-shape model is a useful negative result: forcing one shape restores cosine similarity but does not improve monthly MAE.

The challenger aggregation discrepancy is {result['seasonal_aggregation_max_abs_kgal']:.3g} kgal. Object values remain latent model attributions, not measured turf or irrigation use.
"""
    (core.REPORT/"refinement_report.md").write_text(text,encoding="utf-8")

if __name__=="__main__": main()
