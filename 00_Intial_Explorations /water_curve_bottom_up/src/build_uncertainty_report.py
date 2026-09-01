#!/usr/bin/env python3
"""Build a self-contained HTML report with one uncertainty figure per method."""
from __future__ import annotations

import base64, csv, html, json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import pipeline as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"; REPORT=ROOT/"reports"; FIG=REPORT/"figures"/"uncertainty"
SEED=20260901; N_BOOT=10000
METHODS=[
    ("mean_curve","Training-mean curve","#7f7f7f"),
    ("class_area_nnls","Nonnegative class-area model","#1f77b4"),
    ("proportion_multioutput_rf","Proportional multi-output random forest","#2ca02c"),
    ("landsat_class_month_nnls","Continuous Landsat-conditioned class model","#d95f02"),
    ("annual_class_area_common_shape","Annual class-area magnitude with common shape","#9467bd"),
    ("seasonally_regularized_class_area","Seasonally regularized class-area model","#e7298a"),
]
METHOD_STEPS={
    "mean_curve":[
        "Exclude the held-out parcel.",
        "Average the other parcels’ 12 monthly values.",
        "Use that mean curve as the held-out prediction.",
    ],
    "class_area_nnls":[
        "Measure each land-cover class area in the parcel.",
        "Estimate a nonnegative monthly unit-area curve for each class.",
        "Multiply class rates by object areas and sum them to the parcel.",
    ],
    "proportion_multioutput_rf":[
        "Calculate each parcel’s land-cover proportions.",
        "Fit one random forest to predict all 12 months together.",
        "Apply it to the held-out parcel’s proportions.",
    ],
    "landsat_class_month_nnls":[
        "Attach monthly Landsat NDVI, NDMI, and EVI to each object.",
        "Estimate nonnegative class rates that vary with those values.",
        "Multiply rates by object areas and sum them to the parcel.",
    ],
    "annual_class_area_common_shape":[
        "Estimate annual class-area rates from the training parcels.",
        "Estimate one common monthly shape from those parcels.",
        "Distribute each object’s annual estimate across that shape and sum.",
    ],
    "seasonally_regularized_class_area":[
        "Measure each parcel’s land-cover class areas.",
        "Fit all nonnegative class-month rates jointly while smoothing adjacent months.",
        "Multiply the smoothed rates by object areas and sum them to the parcel.",
    ],
}

def read_predictions():
    data={}; target=None
    with (OUT/"parcel_predictions.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name=r["model_id"]; pid=int(r["fid"])
            obs=np.array([float(r[x]) for x in core.TARGETS]); pred=np.array([float(r[f"held_{m:02d}"]) for m in range(1,13)])
            data.setdefault(name,[]).append((pid,obs,pred))
    with (OUT/"refinement_parcel_predictions.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name=r["model_id"]; pid=int(r["parcel_fid"])
            obs=np.array([float(r[f"observed_{m:02d}"]) for m in range(1,13)]); pred=np.array([float(r[f"held_{m:02d}"]) for m in range(1,13)])
            data.setdefault(name,[]).append((pid,obs,pred))
    result={}
    for name,rows in data.items():
        rows=sorted(rows); ids=[r[0] for r in rows]; obs=np.array([r[1] for r in rows]); pred=np.array([r[2] for r in rows])
        if len(rows)!=38 or len(set(ids))!=38: raise ValueError(f"{name}: expected 38 unique held-out parcels")
        result[name]=(ids,obs,pred)
        target=obs if target is None else target
        if not np.allclose(obs,target): raise ValueError(f"{name}: observed targets are not aligned")
    expected={x[0] for x in METHODS}
    if set(result)!=expected: raise ValueError(f"method mismatch: {set(result)^expected}")
    return result

def bootstrap_curve(obs,pred,seed):
    """Paired parcel bootstrap distributions of cohort mean curves and errors."""
    rng=np.random.default_rng(seed); n=len(obs); idx=rng.integers(0,n,size=(N_BOOT,n))
    pred_mean=pred[idx].mean(axis=1); obs_mean=obs[idx].mean(axis=1)
    diff_mae=np.mean(np.abs(pred[idx]-obs[idx]),axis=(1,2))
    return {
        "prediction_quantiles":np.quantile(pred_mean,[.025,.5,.975],axis=0),
        "observed_quantiles":np.quantile(obs_mean,[.025,.5,.975],axis=0),
        "monthly_mae_quantiles":np.quantile(diff_mae,[.025,.5,.975]),
    }

def figure(name,label,color,obs,pred,boot):
    months=np.arange(1,13); oq=boot["observed_quantiles"]; pq=boot["prediction_quantiles"]
    fig,ax=plt.subplots(figsize=(9.2,5.2))
    ax.fill_between(months,pq[0],pq[2],color=color,alpha=.22,label="95% paired parcel-bootstrap band")
    ax.plot(months,pred.mean(0),color=color,lw=2.6,marker="o",ms=4,label="Mean held-out prediction")
    ax.plot(months,obs.mean(0),color="black",lw=2.6,marker="o",ms=4,label="Mean observed")
    ax.set(title=label,xlabel="Month of 2021",ylabel="Water use (thousand gallons/month)",xticks=months)
    ax.grid(alpha=.22); ax.legend(frameon=False,loc="upper left"); ax.set_xlim(.7,12.3); ax.set_ylim(bottom=0)
    ax.text(.99,.02,"Band: uncertainty in the mean held-out prediction across parcels\n—not an individual-parcel prediction interval",ha="right",va="bottom",transform=ax.transAxes,fontsize=8,color="#444")
    fig.tight_layout(); path=FIG/f"{name}.png"; fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return path

def representative_indices(obs):
    """Choose fixed low, median, and high annual-use parcels for every method."""
    order=np.argsort(obs.sum(1)); n=len(order)
    return [int(order[n//6]),int(order[n//2]),int(order[(5*n)//6])]

def parcel_figure(name,label,color,ids,obs,pred,indices):
    """Show genuine LOPO predictions with leave-one-out residual-calibrated bands."""
    months=np.arange(1,13); residual=np.abs(pred-obs)
    fig,axes=plt.subplots(1,3,figsize=(15.2,5.25),sharex=True)
    levels=["Lower-use parcel","Middle-use parcel","Higher-use parcel"]
    for ax,i,level in zip(axes,indices,levels):
        calibration=np.delete(residual,i,axis=0)
        # Higher empirical quantile gives finite-sample conservative 90% bands.
        q=np.quantile(calibration,.90,axis=0,method="higher")
        lower=np.maximum(0,pred[i]-q); upper=pred[i]+q
        ax.fill_between(months,lower,upper,color=color,alpha=.22,label="90% LOO residual band")
        ax.plot(months,pred[i],color=color,lw=2.3,marker="o",ms=3.5,label="Held-out prediction")
        ax.plot(months,obs[i],color="black",lw=2.3,marker="o",ms=3.5,label="Observed")
        ax.set_title(f"{level}\nParcel FID {ids[i]} · observed annual {obs[i].sum():.1f} kgal",fontsize=10)
        ax.set_xticks(months); ax.set_xlim(.7,12.3); ax.set_ylim(bottom=0); ax.grid(alpha=.22); ax.set_xlabel("Month")
    axes[0].set_ylabel("Water use (thousand gallons/month)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
    fig.suptitle(f"{label}: actual held-out parcel examples",fontsize=16,fontweight="bold")
    fig.text(.5,.105,"Each colored curve is a genuine LOPO prediction. Its band uses month-specific absolute residuals from the other 37 held-out parcels.",ha="center",fontsize=9,color="#444")
    fig.tight_layout(rect=(0,.17,1,.92)); path=FIG/f"{name}_held_out_parcels.png"; fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return path

def img_uri(path): return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode("ascii")

def metric_lookup():
    rows=[]
    for path in (OUT/"model_comparison.csv",OUT/"refinement_model_comparison.csv"):
        with path.open(newline="",encoding="utf-8") as f: rows.extend(csv.DictReader(f))
    return {r["model_id"]:r for r in rows}

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); FIG.mkdir(parents=True,exist_ok=True)
    data=read_predictions(); metrics=metric_lookup(); sections=[]; summary=[]
    reference_obs=data[METHODS[0][0]][1]; selected=representative_indices(reference_obs)
    for i,(name,label,color) in enumerate(METHODS):
        ids,obs,pred=data[name]; boot=bootstrap_curve(obs,pred,SEED+i); path=figure(name,label,color,obs,pred,boot); parcel_path=parcel_figure(name,label,color,ids,obs,pred,selected); m=metrics[name]; q=boot["monthly_mae_quantiles"]
        summary.append({"model_id":name,"label":label,"monthly_mae":float(m["monthly_mae_kgal"]),"monthly_rmse":float(m["monthly_rmse_kgal"]),"cosine":float(m["mean_cosine_similarity"]),"annual_mae":float(m["annual_mae_kgal"]),"bootstrap_mae_p025":float(q[0]),"bootstrap_mae_p975":float(q[2]),"aggregate_figure":str(path.relative_to(ROOT)),"held_out_parcel_figure":str(parcel_path.relative_to(ROOT)),"held_out_parcel_fids":[ids[j] for j in selected]})
        steps="".join(f"<li>{html.escape(x)}</li>" for x in METHOD_STEPS[name])
        sections.append(f'''<section><h2>{html.escape(label)}</h2><div class="methodid">Method ID: <code>{html.escape(name)}</code></div><div class="how"><strong>How it works</strong><ol>{steps}</ol></div><div class="metricline">LOPO monthly MAE: <strong>{float(m['monthly_mae_kgal']):.3f}</strong> kgal &nbsp;·&nbsp; RMSE: <strong>{float(m['monthly_rmse_kgal']):.3f}</strong> &nbsp;·&nbsp; mean cosine: <strong>{float(m['mean_cosine_similarity']):.3f}</strong> &nbsp;·&nbsp; annual MAE: <strong>{float(m['annual_mae_kgal']):.3f}</strong> kgal</div><h3>Average held-out curve</h3><img src="{img_uri(path)}" alt="Observed and predicted monthly curve with bootstrap uncertainty for {html.escape(label)}"><p>The colored band is the 95% paired parcel-bootstrap interval for the method’s <em>mean held-out monthly prediction</em>. The bootstrap resamples the same 38 observed/predicted parcel pairs, preserving their 12-month curves. The bootstrap interval for overall monthly MAE is {q[0]:.3f}–{q[2]:.3f} kgal.</p><h3>Actual held-out parcel examples</h3><img src="{img_uri(parcel_path)}" alt="Three actual held-out parcel predictions with residual uncertainty bands for {html.escape(label)}"><p>These are the same lower-, middle-, and higher-use parcels for every method. Each colored curve is the parcel’s genuine LOPO prediction. For each displayed parcel, the 90% band is calibrated month by month from absolute held-out residuals of the other 37 parcels, excluding the displayed parcel itself. It is an empirical predictive-error band, not a refitted parameter interval.</p></section>''')
    rows="".join(f"<tr><td>{html.escape(x['label'])}</td><td>{x['monthly_mae']:.3f}</td><td>{x['monthly_rmse']:.3f}</td><td>{x['cosine']:.3f}</td><td>{x['annual_mae']:.3f}</td><td>{x['bootstrap_mae_p025']:.3f}–{x['bootstrap_mae_p975']:.3f}</td></tr>" for x in summary)
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Monthly water-use model uncertainty</title><style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1180px;margin:0 auto;padding:32px;color:#222;line-height:1.5}} h1{{margin-bottom:.2rem}} h3{{margin-top:28px}} .sub{{color:#555;margin-top:0}} .note{{background:#f4f6f8;border-left:5px solid #496a86;padding:14px 18px;margin:24px 0}} table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} section{{margin:48px 0 62px;border-top:1px solid #bbb;padding-top:28px}} img{{display:block;width:100%;height:auto;margin:14px 0}} .methodid{{font-size:.9rem;color:#555;margin:-8px 0 8px}} .how{{background:#fafafa;border:1px solid #e2e2e2;padding:9px 14px;margin:10px 0 14px}} .how ol{{margin:5px 0 0 22px;padding:0}} .how li{{margin:1px 0}} .metricline{{color:#444;font-size:.94rem}} code{{background:#f2f2f2;padding:2px 4px}} footer{{border-top:1px solid #ddd;margin-top:50px;padding-top:18px;color:#666;font-size:.88rem}}</style></head><body>
<h1>Held-out monthly water-use curves and uncertainty</h1><p class="sub">Six models · 38 parcels · raw January–December 2021 water use · units: thousand gallons/month</p>
<div class="note"><strong>How to read the figures.</strong> Every method has two clearly labeled figures. The first summarizes the average curve: black is mean observed, colored is mean LOPO prediction, and shading is a 95% paired parcel-bootstrap confidence interval for the average prediction. The second shows three actual held-out parcels: black is that parcel’s observed curve, colored is its genuine LOPO prediction, and shading is a 90% leave-one-out residual-calibrated predictive-error band. Neither band includes land-cover classification, meter, or causal-attribution uncertainty.</div>
<h2>Comparison</h2><table><thead><tr><th>Method</th><th>Monthly MAE</th><th>RMSE</th><th>Mean cosine</th><th>Annual MAE</th><th>Bootstrap MAE 95%</th></tr></thead><tbody>{rows}</tbody></table>
<p>The seasonally regularized class-area model has the best point metrics, but its improvement over the original class-area model remains small and the paired difference interval reported in the refinement analysis includes zero. The figures should therefore be read as comparative uncertainty evidence, not as proof of a decisive ranking.</p>
{''.join(sections)}
<footer>Generated deterministically by <code>src/build_uncertainty_report.py</code> from saved LOPO predictions produced from the three authoritative input datasets. Bootstrap seed: {SEED}; draws: {N_BOOT:,}.</footer></body></html>'''
    path=REPORT/"water_use_uncertainty_report.html"; path.write_text(document,encoding="utf-8")
    (OUT/"uncertainty_figure_summary.json").write_text(json.dumps({"bootstrap":{"draws":N_BOOT,"seed":SEED,"unit":"paired parcels","estimand":"mean held-out monthly prediction curve"},"methods":summary},indent=2),encoding="utf-8")
    core.build_manifest(); print(f"Wrote {path} with {len(METHODS)} method sections and {2*len(METHODS)} embedded figures")

if __name__=="__main__": main()
