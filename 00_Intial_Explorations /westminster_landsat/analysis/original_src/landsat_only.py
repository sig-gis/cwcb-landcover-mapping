#!/usr/bin/env python3
"""Class-free Landsat-only prediction experiments."""
from __future__ import annotations
import base64,csv,html,json,math,os
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src import pipeline as core
from src import refinement

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; REPORT=ROOT/"reports"; FIG=REPORT/"figures"/"landsat_only"
SEED=20260903; ALPHAS=[.1,1.,10.,100.,1000.]
METHODS=[
    ("landsat_only_parcel_ridge","Parcel trajectory ridge","#6a3d9a"),
    ("landsat_only_parcel_rf","Parcel trajectory random forest","#33a02c"),
    ("landsat_only_bottomup_nonnegative","Bottom-up nonnegative Landsat response","#1f78b4"),
    ("landsat_only_bottomup_signed","Bottom-up signed Landsat response","#e31a1c"),
]

def parcel_trajectory(fragments,n):
    x=np.zeros((n,12,3)); area=np.zeros(n)
    for f in fragments:
        a=f["area_m2"]; x[f["parcel"]]+=a*f["sat"]; area[f["parcel"]]+=a
    return (x/area[:,None,None]).reshape(n,-1)

def scale(train_x,x):
    mean=train_x.mean(0); sd=train_x.std(0); sd[sd==0]=1; return (x-mean)/sd

def choose_parcel_ridge(x,y,train):
    scores={}
    for a in ALPHAS:
        e=[]
        for val in train:
            tr=train[train!=val]; z=scale(x[tr],x); model=Ridge(alpha=a); model.fit(z[tr],y[tr]); e.extend((model.predict(z[[val]])[0]-y[val])**2)
        scores[a]=np.mean(e)
    return min(scores,key=scores.get)

def lopo_parcel_ridge(x,y):
    p=np.zeros_like(y); selected=[]
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); a=choose_parcel_ridge(x,y,tr); selected.append(a); z=scale(x[tr],x); model=Ridge(alpha=a); model.fit(z[tr],y[tr]); p[held]=model.predict(z[[held]])[0]
    a=choose_parcel_ridge(x,y,np.arange(len(y))); z=scale(x,x); model=Ridge(alpha=a).fit(z,y)
    return p,model.predict(z),a,selected

def lopo_rf(x,y):
    p=np.zeros_like(y)
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); model=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,bootstrap=True,random_state=SEED+held,n_jobs=1); model.fit(x[tr],y[tr]); p[held]=model.predict(x[[held]])[0]
    model=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,bootstrap=True,random_state=SEED,n_jobs=1).fit(x,y)
    return p,model.predict(x)

def bottomup_design(fragments,n,month,train):
    lo,span=core.sat_scaler(fragments,set(map(int,train)),month); d=np.zeros((n,4))
    for f in fragments:
        z=np.clip((f["sat"][month]-lo)/span,0,1); d[f["parcel"]]+=f["area_m2"]/core.AREA_UNIT*np.r_[1,z]
    return d,lo,span

def signed_solve(d,y,a):
    return np.linalg.lstsq(np.vstack([d,math.sqrt(a)*np.eye(d.shape[1])]),np.r_[y,np.zeros(d.shape[1])],rcond=None)[0]

def choose_bottomup(ds,y,train,signed):
    solve=signed_solve if signed else core.ridge_nnls; scores={}
    for a in ALPHAS:
        e=[]
        for val in train:
            tr=train[train!=val]
            for m,d in enumerate(ds): e.append((d[val]@solve(d[tr],y[tr,m],a)-y[val,m])**2)
        scores[a]=np.mean(e)
    return min(scores,key=scores.get)

def lopo_bottomup(fragments,n,y,signed):
    p=np.zeros_like(y); selected=[]
    for held in range(n):
        tr=np.delete(np.arange(n),held); ds=[bottomup_design(fragments,n,m,tr)[0] for m in range(12)]; a=choose_bottomup(ds,y,tr,signed); selected.append(a); solve=signed_solve if signed else core.ridge_nnls
        for m,d in enumerate(ds): p[held,m]=d[held]@solve(d[tr],y[tr,m],a)
    ds=[]; scales=[]
    for m in range(12): d,lo,span=bottomup_design(fragments,n,m,np.arange(n)); ds.append(d); scales.append((lo,span))
    a=choose_bottomup(ds,y,np.arange(n),signed); solve=signed_solve if signed else core.ridge_nnls; coef=np.column_stack([solve(ds[m],y[:,m],a) for m in range(12)])
    return p,coef,ds,scales,a,selected

def object_rates(objects,coef,scales):
    r=np.zeros((len(objects),12))
    for i,o in enumerate(objects):
        for m,(lo,span) in enumerate(scales): r[i,m]=np.r_[1,np.clip((o["sat"][m]-lo)/span,0,1)]@coef[:,m]
    return r

def aggregate(fragments,rates,n):
    p=np.zeros((n,12))
    for f in fragments: p[f["parcel"]]+=f["area_m2"]/core.AREA_UNIT*rates[f["object_i"]]
    return p

def uri(p): return "data:image/png;base64,"+base64.b64encode(p.read_bytes()).decode()

def figures(ids,y,preds):
    FIG.mkdir(parents=True,exist_ok=True); result={}; months=np.arange(1,13); order=np.argsort(y.sum(1)); selected=[order[len(y)//6],order[len(y)//2],order[5*len(y)//6]]
    for name,label,color in METHODS:
        p=preds[name]; rng=np.random.default_rng(SEED); ix=rng.integers(0,len(y),(10000,len(y))); q=np.quantile(p[ix].mean(1),[.025,.975],axis=0)
        fig,ax=plt.subplots(figsize=(9.2,5.2)); ax.fill_between(months,q[0],q[1],color=color,alpha=.22,label="95% parcel-bootstrap band"); ax.plot(months,p.mean(0),color=color,lw=2.5,marker="o",label="Mean held-out prediction"); ax.plot(months,y.mean(0),"k-o",lw=2.5,label="Mean observed"); ax.set(title=label,xlabel="Month",ylabel="Thousand gallons/month",xticks=months); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); a=FIG/f"{name}.png"; fig.savefig(a,dpi=170); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,5.1)); residual=np.abs(p-y)
        for ax,j,level in zip(axes,selected,["Lower use","Middle use","Higher use"]):
            band=np.quantile(np.delete(residual,j,axis=0),.9,axis=0,method="higher"); ax.fill_between(months,np.maximum(0,p[j]-band),p[j]+band,color=color,alpha=.22,label="90% LOO residual band"); ax.plot(months,p[j],color=color,lw=2.2,marker="o",label="Held-out prediction"); ax.plot(months,y[j],"k-o",lw=2.2,label="Observed"); ax.set_title(f"{level} · parcel FID {ids[j]}"); ax.set_xticks(months); ax.grid(alpha=.2); ax.set_xlabel("Month"); ax.set_ylim(bottom=0)
        axes[0].set_ylabel("Thousand gallons/month"); h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,frameon=False); fig.suptitle(f"{label}: actual held-out parcels",fontweight="bold",fontsize=15); fig.tight_layout(rect=(0,.1,1,.94)); b=FIG/f"{name}_parcels.png"; fig.savefig(b,dpi=170); plt.close(fig); result[name]=(a,b)
    return result

def make_report(metrics,paired,paths,audit):
    rows="".join(f"<tr><td>{html.escape(x['model_id'])}</td><td>{x['monthly_mae_kgal']:.3f}</td><td>{x['monthly_rmse_kgal']:.3f}</td><td>{x['mean_cosine_similarity']:.3f}</td><td>{x['annual_mae_kgal']:.3f}</td></tr>" for x in metrics)
    sections=[]
    descriptions={
      "landsat_only_parcel_ridge":"Area-weight monthly NDVI/NDMI/EVI to each parcel, regularize a linear 36-feature mapping, and predict all months.",
      "landsat_only_parcel_rf":"Area-weight monthly NDVI/NDMI/EVI to each parcel and use one multi-output forest for all months.",
      "landsat_only_bottomup_nonnegative":"Estimate one nonnegative monthly unit-area response to continuous Landsat values, multiply by object area, and sum.",
      "landsat_only_bottomup_signed":"Use the same object equation but allow Landsat-conditioned unit-area contributions to reduce metered demand.",
    }
    for name,label,_ in METHODS:
        a,b=paths[name]; sections.append(f"<section><h2>{label}</h2><p>{descriptions[name]}</p><img src='{uri(a)}'><img src='{uri(b)}'></section>")
    by={x["model_id"]:x for x in metrics}; best=min([x for x in metrics if x['model_id'].startswith('landsat_only')],key=lambda x:x['monthly_mae_kgal']); mean=by["mean_curve"]; land=by["seasonally_regularized_class_area"]
    mean_pair=next(x for x in paired if x["model_a"]=="mean_curve" and x["model_b"]==best["model_id"]); land_pair=next(x for x in paired if x["model_a"]=="seasonally_regularized_class_area" and x["model_b"]==best["model_id"])
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Landsat-only water-use experiment</title><style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:32px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#f4f6f8;border-left:5px solid #6a3d9a;padding:14px}}section{{margin:40px 0;border-top:1px solid #ccc;padding-top:22px}}img{{width:100%}}</style></head><body><h1>How close can Landsat-only models get?</h1><div class="note">No land-cover class labels, class areas, or class proportions enter these four models. Landsat remains contextual ~30 m data shared by many smaller aerial objects. Bottom-up methods use object area only to convert a satellite-conditioned unit rate into a contribution.</div><h2>Held-out comparison</h2><table><tr><th>Model</th><th>Monthly MAE</th><th>RMSE</th><th>Cosine</th><th>Annual MAE</th></tr>{rows}</table><p><strong>Best Landsat-only point result:</strong> {best['model_id']} with monthly MAE {best['monthly_mae_kgal']:.3f} kgal. It improves on the mean curve by only {mean['monthly_mae_kgal']-best['monthly_mae_kgal']:.3f} kgal; the paired 95% interval is [{mean_pair['bootstrap_p025']:.3f}, {mean_pair['bootstrap_p975']:.3f}] and includes zero. It trails the seasonally regularized land-cover model by {best['monthly_mae_kgal']-land['monthly_mae_kgal']:.3f} kgal; the paired interval for land-cover minus Landsat is [{land_pair['bootstrap_p025']:.3f}, {land_pair['bootstrap_p975']:.3f}]. Landsat-only cosine similarity remains close to the reference models, so it recovers the broad seasonal pattern better than parcel-specific magnitude. All preprocessing and tuning are confined to outer-training parcels.</p>{''.join(sections)}<h2>Satellite support</h2><p>{audit['used_pixels']} Landsat supports serve 13,019 aerial objects; shared pixels are preserved. Variables are monthly 2021 NDVI, NDMI, and EVI after Collection 2 scaling and QA filtering.</p></body></html>'''; (REPORT/"landsat_only_report.html").write_text(text,encoding="utf-8")

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); parcels,objects,fragments,classes,_=core.load_spatial(); loc,x,obs,audit=core.landsat_data(objects)
    for i,o in enumerate(objects): o["sat"]=x[i]; o["landsat_id"]=int(loc[i])
    core.attach_sat(fragments,objects,loc,x); y=np.array([p["y"] for p in parcels]); ids=[p["fid"] for p in parcels]; trajectory=parcel_trajectory(fragments,len(y))
    pr,pr_full,pr_a,pr_sel=lopo_parcel_ridge(trajectory,y); rf,rf_full=lopo_rf(trajectory,y); bn,bncoef,bnds,bnscale,bna,bnsel=lopo_bottomup(fragments,len(y),y,False); bs,bscoef,bsds,bsscale,bsa,bssel=lopo_bottomup(fragments,len(y),y,True)
    rn=object_rates(objects,bncoef,bnscale); rs=object_rates(objects,bscoef,bsscale); invn=float(np.max(abs(aggregate(fragments,rn,len(y))-np.column_stack([bn_ds@bncoef[:,m] for m,bn_ds in enumerate(bnds)])))); invs=float(np.max(abs(aggregate(fragments,rs,len(y))-np.column_stack([bs_ds@bscoef[:,m] for m,bs_ds in enumerate(bsds)]))))
    mean=core.mean_lopo(y); d=core.base_matrix(fragments,len(y),len(classes)); lc,_,_,_,_=core.lopo_simple(d,y); seasonal,_,_,_,_,_=refinement.lopo_joint(d,y)
    preds={"mean_curve":mean,"class_area_nnls":lc,"seasonally_regularized_class_area":seasonal,"landsat_only_parcel_ridge":pr,"landsat_only_parcel_rf":rf,"landsat_only_bottomup_nonnegative":bn,"landsat_only_bottomup_signed":bs}; metrics=[core.metrics(k,y,v) for k,v in preds.items()]; paired=core.paired_comparisons(y,preds); core.write_csv(OUT/"landsat_only_model_comparison.csv",metrics); core.write_csv(OUT/"landsat_only_paired_differences.csv",paired)
    rows=[]
    for name in [x[0] for x in METHODS]:
        for i,p in enumerate(parcels):
            r={"parcel_fid":p["fid"],"model_id":name,"validation":"LOPO"}
            for m in range(12): r[f"observed_{m+1:02d}"]=y[i,m]; r[f"held_{m+1:02d}"]=preds[name][i,m]
            rows.append(r)
    core.write_csv(OUT/"landsat_only_held_out_predictions.csv",rows)
    result={"target":core.TARGETS,"predictors":"monthly 2021 NDVI, NDMI, EVI only; no land-cover labels/proportions","metrics":metrics,"paired":paired,"settings":{"parcel_ridge_alpha":pr_a,"bottomup_nonnegative_alpha":bna,"bottomup_signed_alpha":bsa},"aggregation":{"nonnegative_max_abs_kgal":invn,"signed_max_abs_kgal":invs},"satellite_audit":audit}; (OUT/"landsat_only_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); paths=figures(ids,y,{k:preds[k] for k,_,_ in METHODS}); make_report(metrics,paired,paths,audit); core.build_manifest(); print(json.dumps({"metrics":metrics,"aggregation":result["aggregation"]},indent=2))

if __name__=="__main__": main()
