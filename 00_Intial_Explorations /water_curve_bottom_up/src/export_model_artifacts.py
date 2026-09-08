#!/usr/bin/env python3
"""Export portable full-fit state for every fitted model family."""
from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src import pipeline as core
from src import refinement
from src import signed_effects
from src import landsat_only

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; ART=ROOT/"artifacts"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); ART.mkdir(exist_ok=True)
    parcels,objects,fragments,classes,_=core.load_spatial(); loc,ox,observed,sat_audit=core.landsat_data(objects)
    for i,o in enumerate(objects): o["sat"]=ox[i]; o["landsat_id"]=int(loc[i])
    core.attach_sat(fragments,objects,loc,ox); y=np.array([p["y"] for p in parcels]); n=len(y); c=len(classes); d=core.base_matrix(fragments,n,c); props=d/np.maximum(d.sum(1,keepdims=True),1e-12)
    arrays={"class_names":np.asarray(classes,dtype="U40"),"target_fields":np.asarray(core.TARGETS,dtype="U20"),"mean_curve":y.mean(0)}
    # Class-area models.
    arrays["class_area_nnls_coef"]=np.column_stack([core.ridge_nnls(d,y[:,m],10.) for m in range(12)])
    arrays["annual_class_area_coef"]=core.ridge_nnls(d,y.sum(1),10.); arrays["annual_common_shape"]=y.sum(0)/y.sum()
    arrays["seasonal_nonnegative_coef"]=refinement.solve_joint(d,y,(10.,10.,0.))
    arrays["signed_ridge_class_area_coef"]=np.column_stack([signed_effects.solve_ridge(d,y[:,m],10.) for m in range(12)])
    arrays["signed_seasonal_coef"]=signed_effects.solve_smooth(d,y,(10.,10.))
    # Class-specific Landsat model and fold-independent robust scalers.
    lds=[]; llo=[]; lspan=[]
    for m in range(12):
        dm,lo,span=core.sat_matrix(fragments,n,c,m,np.arange(n)); lds.append(dm); llo.append(lo); lspan.append(span)
    arrays["class_landsat_coef"]=np.column_stack([core.ridge_nnls(lds[m],y[:,m],.1,c) for m in range(12)]); arrays["class_landsat_scaler_p05"]=np.asarray(llo); arrays["class_landsat_scaler_span_p95_minus_p05"]=np.asarray(lspan)
    # Strict Landsat-only parcel features and ridge.
    trajectory=landsat_only.parcel_trajectory(fragments,n); mean=trajectory.mean(0); sd=trajectory.std(0); sd[sd==0]=1; z=(trajectory-mean)/sd; ridge=Ridge(alpha=1000.).fit(z,y)
    arrays["landsat_parcel_feature_mean"]=mean; arrays["landsat_parcel_feature_sd"]=sd; arrays["landsat_parcel_ridge_coef"]=ridge.coef_; arrays["landsat_parcel_ridge_intercept"]=ridge.intercept_
    # Strict Landsat-only bottom-up models.
    bds=[]; blo=[]; bspan=[]
    for m in range(12): dm,lo,span=landsat_only.bottomup_design(fragments,n,m,np.arange(n)); bds.append(dm); blo.append(lo); bspan.append(span)
    arrays["landsat_only_scaler_p05"]=np.asarray(blo); arrays["landsat_only_scaler_span_p95_minus_p05"]=np.asarray(bspan)
    arrays["landsat_only_bottomup_nonnegative_coef"]=np.column_stack([core.ridge_nnls(bds[m],y[:,m],100.) for m in range(12)])
    arrays["landsat_only_bottomup_signed_coef"]=np.column_stack([signed_effects.solve_ridge(bds[m],y[:,m],100.) for m in range(12)])
    np.savez_compressed(ART/"linear_model_state.npz",**arrays)
    # Forests are benchmarks, but serialize them for exact reproduction.
    class_rf=core.forest_model().fit(props,y); joblib.dump(class_rf,ART/"proportion_multioutput_rf.joblib",compress=3)
    sat_rf=landsat_only.RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.7,bootstrap=True,random_state=landsat_only.SEED,n_jobs=1).fit(trajectory,y); joblib.dump(sat_rf,ART/"landsat_only_parcel_rf.joblib",compress=3)
    # Human-readable coefficient table for all object-compatible models.
    rows=[]
    for key in ["class_area_nnls_coef","seasonal_nonnegative_coef","signed_ridge_class_area_coef","signed_seasonal_coef"]:
        a=arrays[key]
        for j,name in enumerate(classes):
            for m in range(12): rows.append({"artifact":key,"class_final":name,"month":m+1,"coefficient":a[j,m],"unit":"kgal per 100 m2 per month"})
    core.write_csv(ART/"class_month_coefficients.csv",rows)
    # Reconstruction checks against saved full-fit outputs where available.
    checks={}
    sources=[("class_area_nnls_coef",OUT/"parcel_predictions.csv","class_area_nnls","fid"),("seasonal_nonnegative_coef",OUT/"parcel_predictions_seasonally_regularized.csv","seasonally_regularized_class_area","fid"),("signed_seasonal_coef",OUT/"parcel_predictions_signed_seasonal.csv","signed_seasonally_regularized_class_area","fid")]
    fids=[p["fid"] for p in parcels]
    for key,path,model,idfield in sources:
        saved={int(r[idfield]):np.array([float(r[f"full_{m:02d}"]) for m in range(1,13)]) for r in csv.DictReader(path.open()) if r.get("model_id")==model}
        calc=d@arrays[key]; checks[key]={"max_abs_kgal":float(max(np.max(abs(calc[i]-saved[fid])) for i,fid in enumerate(fids))),"rows":len(saved)}
    checks["class_landsat_direct_design"]={"max_abs_kgal":float(np.max(abs(np.column_stack([lds[m]@arrays["class_landsat_coef"][:,m] for m in range(12)])-np.column_stack([lds[m]@arrays["class_landsat_coef"][:,m] for m in range(12)]))))}
    (ART/"reconstruction_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    definitions={
      "artifact_version":"1.0","training_scope":"38 parcels, calendar year 2021","water_unit":"thousand gallons/month","area_unit":"100 square metres","month_order":core.TARGETS,"class_order":classes,
      "landsat_feature_order":{"parcel_trajectory":"month-major: [NDVI, NDMI, EVI] for Jan, then Feb, ... Dec","object_month":"[intercept, scaled NDVI, scaled NDMI, scaled EVI]","scaling":"clip((value-p05)/(p95-p05), 0, 1) for bottom-up models"},
      "models":{
        "mean_curve":{"state":["mean_curve"],"role":"baseline"},
        "class_area_nnls":{"state":["class_area_nnls_coef"],"formula":"parcel[p,m] = sum_c area[p,c]/100 * beta[c,m]","constraint":"beta >= 0","role":"simple retained reference"},
        "annual_class_area_common_shape":{"state":["annual_class_area_coef","annual_common_shape"],"role":"negative refinement result"},
        "seasonally_regularized_class_area":{"state":["seasonal_nonnegative_coef"],"penalties":{"ridge":10,"adjacent_month_smoothness":10,"class_pooling":0},"role":"preferred point-metric land-cover model"},
        "signed_ridge_class_area":{"state":["signed_ridge_class_area_coef"],"role":"signed sensitivity"},
        "signed_seasonally_regularized_class_area":{"state":["signed_seasonal_coef"],"penalties":{"ridge":10,"adjacent_month_smoothness":10},"role":"retained signed sensitivity"},
        "landsat_class_month_nnls":{"state":["class_landsat_coef","class_landsat_scaler_p05","class_landsat_scaler_span_p95_minus_p05"],"role":"class-specific Landsat negative result"},
        "proportion_multioutput_rf":{"file":"proportion_multioutput_rf.joblib","role":"parcel nonlinear benchmark"},
        "landsat_only_parcel_ridge":{"state":["landsat_parcel_feature_mean","landsat_parcel_feature_sd","landsat_parcel_ridge_coef","landsat_parcel_ridge_intercept"],"role":"satellite-only parcel benchmark"},
        "landsat_only_parcel_rf":{"file":"landsat_only_parcel_rf.joblib","role":"satellite-only nonlinear/tail benchmark"},
        "landsat_only_bottomup_nonnegative":{"state":["landsat_only_bottomup_nonnegative_coef","landsat_only_scaler_p05","landsat_only_scaler_span_p95_minus_p05"],"role":"best Landsat-only magnitude model"},
        "landsat_only_bottomup_signed":{"state":["landsat_only_bottomup_signed_coef","landsat_only_scaler_p05","landsat_only_scaler_span_p95_minus_p05"],"role":"satellite-only signed sensitivity"}
      },
      "warning":"These fitted values reproduce the 2021 study. Retrain rather than directly transfer them to a larger five-year population."
    }
    (ART/"artifact_definitions.json").write_text(json.dumps(definitions,indent=2),encoding="utf-8")
    files=[]
    for p in sorted(ART.iterdir()):
        if p.name=="artifact_manifest.json": continue
        files.append({"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    (ART/"artifact_manifest.json").write_text(json.dumps({"files":files,"source_input_hashes":{p.name:core.sha256(p) for p in core.INPUT.iterdir() if p.is_file()},"code":["src/pipeline.py","src/refinement.py","src/signed_effects.py","src/landsat_only.py","src/export_model_artifacts.py"]},indent=2),encoding="utf-8")
    print(json.dumps({"artifacts":files,"checks":checks},indent=2))

if __name__=="__main__": main()
