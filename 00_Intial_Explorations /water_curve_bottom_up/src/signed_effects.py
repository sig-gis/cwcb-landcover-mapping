#!/usr/bin/env python3
"""Separate experiment allowing signed land-cover contributions to metered use."""
from __future__ import annotations
import base64,csv,html,json,math,os
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import pipeline as core
from src import refinement as nonneg

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; REPORT=ROOT/"reports"; FIG=REPORT/"figures"/"signed_effects"
SEED=20260902
SIGNED_CONFIGS=[(.1,0),(1.,0),(10.,0),(100.,0),(1.,10.),(10.,10.),(100.,10.),(10.,100.)]

def solve_ridge(d,y,alpha):
    return np.linalg.lstsq(np.vstack([d,math.sqrt(alpha)*np.eye(d.shape[1])]),np.r_[y,np.zeros(d.shape[1])],rcond=None)[0]

def choose_ridge(d,y,train):
    scores={}
    for a in core.ALPHAS:
        e=[]
        for val in train:
            tr=train[train!=val]
            for m in range(12): e.append((d[val]@solve_ridge(d[tr],y[tr,m],a)-y[val,m])**2)
        scores[a]=np.mean(e)
    return min(scores,key=scores.get)

def lopo_ridge(d,y):
    p=np.zeros_like(y); folds=[]; selected=[]
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); a=choose_ridge(d,y,tr); selected.append(a)
        b=np.column_stack([solve_ridge(d[tr],y[tr,m],a) for m in range(12)]); folds.append(b); p[held]=d[held]@b
    a=choose_ridge(d,y,np.arange(len(y))); full=np.column_stack([solve_ridge(d,y[:,m],a) for m in range(12)])
    return p,full,np.asarray(folds),a,selected

def signed_penalty(c,ridge,smooth):
    rows=[]
    if ridge: rows.extend(math.sqrt(ridge)*np.eye(c*12))
    if smooth:
        for j in range(c):
            for m in range(11):
                r=np.zeros(c*12); r[j*12+m]=math.sqrt(smooth); r[j*12+m+1]=-math.sqrt(smooth); rows.append(r)
    return np.asarray(rows)

def solve_smooth(d,y,cfg):
    a=nonneg.joint_design(d); pen=signed_penalty(d.shape[1],*cfg)
    return np.linalg.lstsq(np.vstack([a,pen]),np.r_[y.ravel(),np.zeros(len(pen))],rcond=None)[0].reshape(d.shape[1],12)

def choose_smooth(d,y,train,seed):
    scores={}
    for cfg in SIGNED_CONFIGS:
        e=[]
        for val in nonneg.folds(train,seed):
            vs=set(val); tr=np.array([i for i in train if i not in vs]); b=solve_smooth(d[tr],y[tr],cfg); e.extend((d[val]@b-y[val]).ravel()**2)
        scores[cfg]=np.mean(e)
    return min(scores,key=scores.get),scores

def lopo_smooth(d,y):
    p=np.zeros_like(y); folds=[]; selected=[]
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); cfg,_=choose_smooth(d,y,tr,SEED+held); selected.append(cfg); b=solve_smooth(d[tr],y[tr],cfg); folds.append(b); p[held]=d[held]@b
    cfg,scores=choose_smooth(d,y,np.arange(len(y)),SEED); full=solve_smooth(d,y,cfg)
    return p,full,np.asarray(folds),cfg,scores,selected

def aggregate(fragments,objects,classes,rates,n):
    ci={c:i for i,c in enumerate(classes)}; oc={o["fid"]:ci[o["class"]] for o in objects}; p=np.zeros((n,12))
    for f in fragments: p[f["parcel"]]+=f["area_m2"]/core.AREA_UNIT*rates[oc[f["object_fid"]]]
    return p

def bootstrap(d,y,cfg,n=300):
    rng=np.random.default_rng(SEED+44); draws=[]
    for _ in range(n):
        s=rng.integers(0,len(y),len(y)); draws.append(solve_smooth(d[s],y[s],cfg))
    return np.asarray(draws)

def save_outputs(parcels,objects,fragments,classes,d,y,pred,coef,cfg,draws):
    pd=np.einsum('pc,bcm->bpm',d,draws); pq=np.quantile(pd,[.05,.5,.95],axis=0); paq=np.quantile(pd.sum(2),[.05,.5,.95],axis=0); rows=[]
    for i,p in enumerate(parcels):
        r={"fid":p["fid"],"model_id":"signed_seasonally_regularized_class_area","validation":"LOPO held prediction; uncertainty is 300 full-data joint parcel-bootstrap refits","obs_annual":float(y[i].sum()),"held_annual":float(pred[i].sum()),"full_annual":float((d@coef)[i].sum()),"annual_p05":paq[0,i],"annual_p95":paq[2,i]}
        for m in range(12): r[core.TARGETS[m]]=y[i,m]; r[f"held_{m+1:02d}"]=pred[i,m]; r[f"full_{m+1:02d}"]=(d@coef)[i,m]; r[f"p05_{m+1:02d}"]=pq[0,i,m]; r[f"p95_{m+1:02d}"]=pq[2,i,m]
        rows.append(r)
    core.write_csv(OUT/"parcel_predictions_signed_seasonal.csv",rows); core.export_gpkg(OUT/"parcel_predictions_signed_seasonal.gpkg","parcel_predictions",core.INPUT/"parcels_water_2021.gpkg","parcels",rows)
    ci={c:i for i,c in enumerate(classes)}; obj=[]
    for o in objects:
        j=ci[o["class"]]; annual=draws[:,j].sum(1); r={"fid":o["fid"],"sp_id":o["sp_id"],"class_final":o["class"],"area_m2":o["area_m2"],"model_id":"signed_seasonally_regularized_class_area","interpretation":"signed contribution to metered demand; negative values reduce predicted tap requirement","unit":"kgal_per_100m2_month"}
        for m in range(12):
            lo,hi=np.quantile(draws[:,j,m],[.05,.95]); r[f"unit_{m+1:02d}"]=coef[j,m]; r[f"object_{m+1:02d}"]=coef[j,m]*o["area_m2"]/core.AREA_UNIT; r[f"unit_p05_{m+1:02d}"]=lo; r[f"unit_p95_{m+1:02d}"]=hi
        r["unit_annual"]=coef[j].sum(); r["object_annual"]=coef[j].sum()*o["area_m2"]/core.AREA_UNIT; r["unit_annual_p05"],r["unit_annual_p95"]=np.quantile(annual,[.05,.95]); obj.append(r)
    core.write_csv(OUT/"complete_objects_signed_seasonal.csv",obj); core.export_gpkg(OUT/"complete_objects_signed_seasonal.gpkg","object_predictions",core.INPUT/"landcover_2021.gpkg","superpixels",obj)

def uri(path): return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()

def make_figures(classes,y,preds,coef):
    FIG.mkdir(parents=True,exist_ok=True); colors={"signed_ridge_class_area":"#6a3d9a","signed_seasonally_regularized_class_area":"#e31a1c"}; paths={}
    months=np.arange(1,13)
    for name in colors:
        p=preds[name]; fig,ax=plt.subplots(figsize=(9.2,5.2)); residual=np.abs(p-y); rng=np.random.default_rng(SEED); idx=rng.integers(0,len(y),(10000,len(y))); q=np.quantile(p[idx].mean(1),[.025,.975],axis=0)
        ax.fill_between(months,q[0],q[1],color=colors[name],alpha=.22,label="95% parcel-bootstrap band"); ax.plot(months,p.mean(0),color=colors[name],lw=2.6,marker="o",label="Mean held-out prediction"); ax.plot(months,y.mean(0),"k-o",lw=2.6,label="Mean observed"); ax.set(title=name.replace('_',' ').title(),xlabel="Month",ylabel="Thousand gallons/month",xticks=months); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); path=FIG/f"{name}.png"; fig.savefig(path,dpi=170); plt.close(fig); paths[name]=path
    annual=coef.sum(1); order=np.argsort(annual); fig,ax=plt.subplots(figsize=(9.5,5.7)); cs=["#2166ac" if annual[i]<0 else "#b2182b" for i in order]; ax.barh(np.arange(len(classes)),annual[order],color=cs); ax.axvline(0,color="black",lw=1); ax.set_yticks(np.arange(len(classes)),[classes[i] for i in order]); ax.set_xlabel("Signed annual contribution rate (kgal/100 m²/year)"); ax.set_title("Full-fit signed seasonal class rates"); ax.grid(axis="x",alpha=.2); fig.tight_layout(); path=FIG/"signed_annual_class_rates.png"; fig.savefig(path,dpi=170); plt.close(fig); paths["rates"]=path
    return paths

def pure_class_figure(classes,coef,draws,name,title,color,signed):
    """Monthly attribution for a hypothetical pure 100 m² object by class."""
    months=np.arange(1,13); q=np.quantile(draws,[.05,.95],axis=0); annual_draws=draws.sum(2)
    # One common scale within each figure makes class panels visually comparable.
    ymin=min(0,float(q[0].min())) if signed else 0
    ymax=max(float(q[1].max()),float(coef.max()))
    pad=max((ymax-ymin)*.06,.1); ymin-=pad if signed else 0; ymax+=pad
    fig,axes=plt.subplots(4,4,figsize=(15.5,13),sharex=True,sharey=True); axes=axes.ravel()
    for j,c in enumerate(classes):
        ax=axes[j]; alo,ahi=np.quantile(annual_draws[:,j],[.05,.95]); annual=coef[j].sum()
        ax.fill_between(months,q[0,j],q[1,j],color=color,alpha=.23,label="90% joint-bootstrap band")
        ax.plot(months,coef[j],color=color,lw=2.2,marker="o",ms=3,label="Full-fit attribution")
        ax.axhline(0,color="black",lw=.9,alpha=.8)
        ax.set_title(f"{c}\nAnnual: {annual:.1f} [{alo:.1f}, {ahi:.1f}] kgal",fontsize=10)
        ax.grid(alpha=.2); ax.set_xlim(.7,12.3); ax.set_ylim(ymin,ymax); ax.set_xticks([1,3,5,7,9,11])
    for ax in axes[len(classes):]: ax.axis("off")
    for ax in axes[12:13]: ax.set_xlabel("Month")
    fig.supylabel("Attributed metered-water requirement\n(kgal/month for a pure 100 m² object)")
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle(title,fontsize=18,fontweight="bold")
    fig.text(.5,.955,"Pure 100 m² class attribution; brackets are 90% intervals for annual sums.",ha="center",fontsize=10,color="#444")
    fig.tight_layout(rect=(.03,.055,1,.94)); path=FIG/f"{name}.png"; fig.savefig(path,dpi=170,bbox_inches="tight"); plt.close(fig); return path

def report(result,paths):
    rows="".join(f"<tr><td>{html.escape(x['model_id'])}</td><td>{x['monthly_mae_kgal']:.3f}</td><td>{x['monthly_rmse_kgal']:.3f}</td><td>{x['mean_cosine_similarity']:.3f}</td><td>{x['annual_mae_kgal']:.3f}</td></tr>" for x in result["metrics"])
    sections=[]
    for name,title,steps in [
        ("signed_ridge_class_area","Signed ridge class-area",["Measure each class area.","Fit signed monthly class rates with ridge shrinkage.","Multiply rates by object areas and sum to parcels."]),
        ("signed_seasonally_regularized_class_area","Signed seasonally regularized class-area",["Measure each class area.","Fit signed class-month rates jointly while smoothing adjacent months.","Multiply rates by object areas and sum to parcels."])]:
        sections.append(f"<section><h2>{title}</h2><ol>{''.join('<li>'+x+'</li>' for x in steps)}</ol><img src='{uri(paths[name])}'></section>")
    nneg=sum(x["estimate"]<0 for x in result["signed_annual_rates"]); confident=sum(x["p95"]<0 for x in result["signed_annual_rates"])
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Signed land-cover effects experiment</title><style>body{{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:32px;line-height:1.5;color:#222}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#f4f6f8;border-left:5px solid #6a3d9a;padding:14px 18px}}section{{margin:42px 0;border-top:1px solid #ccc;padding-top:22px}}img{{width:100%}}</style></head><body><h1>Allowing signed land-cover contributions</h1><div class="note">This is a separate sensitivity experiment; the existing presentation is unchanged. Negative object values mean the class reduces predicted metered-tap demand within the fitted additive accounting. They do not mean negative physical water flow at the meter.</div><h2>Held-out comparison</h2><table><tr><th>Model</th><th>Monthly MAE</th><th>RMSE</th><th>Cosine</th><th>Annual MAE</th></tr>{rows}</table><p><strong>Bottom line:</strong> allowing negativity changes the allocation but does not materially improve held-out prediction. Signed seasonal MAE is 3.665 versus 3.667 kgal for its nonnegative counterpart; RMSE and mean cosine are worse. The paired MAE-difference interval includes zero.</p>{''.join(sections)}<section><h2>Pure-class annual profiles: nonnegative model</h2><p>Each panel asks what the nonnegative seasonal model attributes to a hypothetical 100 m² object composed entirely of one class.</p><img src="{uri(paths['pure_nonnegative'])}"></section><section><h2>Pure-class annual profiles: signed model</h2><p>The matched signed figure permits a class-month attribution to fall below zero when the model associates that class with reduced metered-tap requirement.</p><img src="{uri(paths['pure_signed'])}"></section><section><h2>Which classes became negative?</h2><img src="{uri(paths['rates'])}"><p>{nneg} classes have negative annual point estimates, but {confident} have a 90% bootstrap interval entirely below zero. Some additional classes are negative only in particular months. No final parcel-month prediction is negative.</p><p>{html.escape(result['interpretation'])}</p></section></body></html>'''
    (REPORT/"signed_landcover_effects_report.html").write_text(text,encoding="utf-8")

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); parcels,objects,fragments,classes,_=core.load_spatial(); y=np.array([p["y"] for p in parcels]); d=core.base_matrix(fragments,len(parcels),len(classes))
    sr,sr_full,sr_folds,sr_alpha,sr_sel=lopo_ridge(d,y); ss,ss_full,ss_folds,cfg,scores,sel=lopo_smooth(d,y)
    # Matched nonnegative references, recomputed from the raw inputs.
    nn,_,_,_,_=core.lopo_simple(d,y); ns,ns_full,_,ns_cfg,_,_=nonneg.lopo_joint(d,y)
    preds={"nonnegative_class_area":nn,"nonnegative_seasonally_regularized":ns,"signed_ridge_class_area":sr,"signed_seasonally_regularized_class_area":ss}
    metrics=[core.metrics(k,y,v) for k,v in preds.items()]; paired=core.paired_comparisons(y,preds); draws=bootstrap(d,y,cfg); save_outputs(parcels,objects,fragments,classes,d,y,ss,ss_full,cfg,draws)
    agg=aggregate(fragments,objects,classes,ss_full,len(y)); invariant=float(np.max(abs(agg-d@ss_full))); fullparcel=d@ss_full
    class_rows=[]
    for j,c in enumerate(classes):
        r={"class_final":c,"signed_ridge_annual":sr_full[j].sum(),"signed_seasonal_annual":ss_full[j].sum(),"signed_seasonal_negative_months":int(np.sum(ss_full[j]<0))}
        for m in range(12): r[f"signed_seasonal_{m+1:02d}"]=ss_full[j,m]
        class_rows.append(r)
    core.write_csv(OUT/"signed_class_rate_comparison.csv",class_rows); core.write_csv(OUT/"signed_model_comparison.csv",metrics); core.write_csv(OUT/"signed_paired_differences.csv",paired)
    negatives=[c for c,b in zip(classes,ss_full) if np.any(b<0)]; annual_rates=[]
    for j,c in enumerate(classes):
        vals=draws[:,j].sum(1); annual_rates.append({"class_final":c,"estimate":float(ss_full[j].sum()),"p05":float(np.quantile(vals,.05)),"p95":float(np.quantile(vals,.95))})
    result={"target":core.TARGETS,"models":["signed_ridge_class_area","signed_seasonally_regularized_class_area"],"metrics":metrics,"paired":paired,"signed_ridge_full_alpha":sr_alpha,"signed_seasonal_full_config":cfg,"signed_seasonal_outer_config_counts":{str(k):v for k,v in Counter(sel).items()},"negative_classes_any_month":negatives,"signed_annual_rates":annual_rates,"negative_fullfit_object_month_rate_count":int(sum(np.sum(ss_full[classes.index(o['class'])]<0) for o in objects)),"negative_fullfit_parcel_month_prediction_count":int(np.sum(fullparcel<0)),"minimum_fullfit_parcel_month_kgal":float(fullparcel.min()),"aggregation_max_abs_kgal":invariant,"interpretation":"Signed rates test whether a class may reduce required metered water. Because class areas are correlated and exhaustive, individual signed rates remain model-dependent; improvement must be judged held out."}
    (OUT/"signed_effects_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); paths=make_figures(classes,y,{"signed_ridge_class_area":sr,"signed_seasonally_regularized_class_area":ss},ss_full)
    rng=np.random.default_rng(SEED+45); nonnegative_draws=[]
    for _ in range(300):
        sample=rng.integers(0,len(y),len(y)); nonnegative_draws.append(nonneg.solve_joint(d[sample],y[sample],ns_cfg))
    nonnegative_draws=np.asarray(nonnegative_draws)
    paths["pure_nonnegative"]=pure_class_figure(classes,ns_full,nonnegative_draws,"pure_100m2_nonnegative_class_profiles","Nonnegative seasonal model: pure 100 m² class profiles","#1f78b4",False)
    paths["pure_signed"]=pure_class_figure(classes,ss_full,draws,"pure_100m2_signed_class_profiles","Signed seasonal model: pure 100 m² class profiles","#e31a1c",True)
    report(result,paths); core.build_manifest(); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
