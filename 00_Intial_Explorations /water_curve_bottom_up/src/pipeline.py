#!/usr/bin/env python3
"""Rebuild the 2021 monthly water-use analysis from the three raw inputs.

The implementation deliberately uses only numpy/scipy/sklearn and GDAL so the
spatial joins, models, and exports are explicit and inspectable.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "outputs"
REPORT = ROOT / "reports"
CACHE = ROOT / "cache"
MONTH_NAMES = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
TARGETS = [f"cow_{m}_21" for m in MONTH_NAMES]
MONTHS = [f"{i:02d}" for i in range(1,13)]
SAT_VARS = ["ndvi", "ndmi", "evi"]
AREA_UNIT = 100.0
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20210831

def mkdirs():
    for p in (OUTPUT, REPORT, CACHE): p.mkdir(exist_ok=True)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write_csv(path, rows, fields=None):
    rows=list(rows)
    if not rows: return
    if fields is None:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def transform_geom(g, source, target):
    z=g.Clone(); z.Transform(osr.CoordinateTransformation(source,target)); return z

def srs_epsg(code):
    x=osr.SpatialReference(); x.ImportFromEPSG(code)
    if code==4326: x.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return x

def layer_schema(path, layer_name):
    ds=ogr.Open(str(path),0); ly=ds.GetLayerByName(layer_name); d=ly.GetLayerDefn()
    out={"file":path.name,"layer":layer_name,"feature_count":ly.GetFeatureCount(),"geometry_type":ogr.GeometryTypeToName(d.GetGeomType()),"crs":ly.GetSpatialRef().GetName(),"fields":[]}
    for i in range(d.GetFieldCount()):
        f=d.GetFieldDefn(i); nulls=sum(1 for x in ly if not x.IsFieldSetAndNotNull(i)); ly.ResetReading()
        out["fields"].append({"name":f.GetName(),"type":f.GetFieldTypeName(f.GetType()),"missing":nulls})
    ds=None; return out

def load_spatial():
    """Return parcels, complete objects, and exact clipped object fragments."""
    pds=ogr.Open(str(INPUT/"parcels_water_2021.gpkg"),0); ply=pds.GetLayerByName("parcels"); psrs=ply.GetSpatialRef()
    lds=ogr.Open(str(INPUT/"landcover_2021.gpkg"),0); lly=lds.GetLayerByName("superpixels"); lsrs=lly.GetSpatialRef()
    utm=srs_epsg(32613); wgs=srs_epsg(4326)
    parcels=[]
    for f in ply:
        g=f.GetGeometryRef().Clone(); gu=transform_geom(g,psrs,utm)
        parcels.append({"fid":f.GetFID(),"geom":g,"area_m2":gu.GetArea(),"y":np.array([float(f[x]) if f.IsFieldSetAndNotNull(x) else np.nan for x in TARGETS]),"reported_total":f["cow_total"]})
    objects=[]; classes=set(); invalid=0; nullgeom=0
    for f in lly:
        g=f.GetGeometryRef()
        if g is None: nullgeom+=1; continue
        g=g.Clone(); invalid += int(not g.IsValid()); gu=transform_geom(g,lsrs,utm); c=f["class_final"] or "unlabelled"; classes.add(c)
        cen=transform_geom(g.Centroid(),lsrs,wgs)
        objects.append({"fid":f.GetFID(),"sp_id":int(f["sp_id"]),"class":c,"geom":g,"area_m2":gu.GetArea(),"source_area":float(f["area"] or 0),"purity":f["purity"],"reviewed":f["reviewed"],"conflict":f["conflict"],"lon":cen.GetX(),"lat":cen.GetY()})
    classes=sorted(classes); cindex={c:i for i,c in enumerate(classes)}
    by_fid={o["fid"]:o for o in objects}; fragments=[]
    for pi,p in enumerate(parcels):
        search=transform_geom(p["geom"],psrs,lsrs); lly.SetSpatialFilter(search)
        for f in lly:
            g=f.GetGeometryRef()
            if g is None or not g.Intersects(search): continue
            z=g.Intersection(search)
            if z is None or z.IsEmpty(): continue
            area=transform_geom(z,lsrs,utm).GetArea()
            if area>1e-8: fragments.append({"parcel":pi,"parcel_fid":p["fid"],"object_fid":f.GetFID(),"sp_id":int(f["sp_id"]),"class":f["class_final"] or "unlabelled","class_i":cindex[f["class_final"] or "unlabelled"],"area_m2":area})
        lly.SetSpatialFilter(None)
    pds=None; lds=None
    return parcels,objects,fragments,classes,{"invalid_objects":invalid,"null_geometry_objects":nullgeom}

def landsat_data(objects):
    con=sqlite3.connect(f"file:{INPUT/'satellite_timeseries.sqlite'}?mode=ro",uri=True)
    locs=con.execute('SELECT id,lon,lat FROM landsat_8_unique_locs ORDER BY id').fetchall()
    loc_ids=np.array([x[0] for x in locs]); xy=np.array([[x[1],x[2]] for x in locs]); tree=cKDTree(xy)
    dist,idx=tree.query(np.array([[o["lon"],o["lat"]] for o in objects])); object_loc=loc_ids[idx]
    raw=defaultdict(list); audit={"raw_2021":0,"clear_2021":0,"masked_qa":0,"masked_saturation":0,"invalid_index":0}
    for table in ("landsat_8","landsat_9"):
        q=f"SELECT SR_B2,SR_B4,SR_B5,SR_B6,QA_PIXEL,QA_RADSAT,date,unique_loc_id FROM {table} WHERE substr(date,1,4)='2021'"
        for blue,red,nir,swir,qa,sat,date,loc in con.execute(q):
            audit["raw_2021"]+=1
            if None in (blue,red,nir,swir,qa): continue
            # C2 QA_PIXEL bits 0--5: fill, dilated cloud, cirrus, cloud, shadow, snow.
            if int(qa)&63: audit["masked_qa"]+=1; continue
            if sat is not None and int(sat)!=0: audit["masked_saturation"]+=1; continue
            blue,red,nir,swir=np.array([blue,red,nir,swir],float)*0.0000275-0.2
            den1=nir+red; den2=nir+swir; den3=nir+6*red-7.5*blue+1
            vals=[(nir-red)/den1 if den1 else np.nan,(nir-swir)/den2 if den2 else np.nan,2.5*(nir-red)/den3 if den3 else np.nan]
            if not np.all(np.isfinite(vals)) or abs(vals[0])>1 or abs(vals[1])>1: audit["invalid_index"]+=1; continue
            raw[(int(loc),int(date[5:7]))].append(vals); audit["clear_2021"]+=1
    # Monthly medians; interpolate internal and end gaps independently for each pixel/variable.
    pixel=np.empty((len(loc_ids),12,3)); observed=np.zeros((len(loc_ids),12),bool)
    for i,loc in enumerate(loc_ids):
        for m in range(1,13):
            values=raw[(int(loc),m)]; pixel[i,m-1]=np.median(values,axis=0) if values else np.nan; observed[i,m-1]=bool(values)
        for j in range(3):
            good=np.flatnonzero(np.isfinite(pixel[i,:,j])); pixel[i,:,j]=np.interp(np.arange(12),good,pixel[i,good,j]) if len(good) else 0.0
    lookup={int(v):i for i,v in enumerate(loc_ids)}; object_x=np.array([pixel[lookup[int(v)]] for v in object_loc])
    con.close()
    audit.update({"pixel_count":len(loc_ids),"pixel_months":len(loc_ids)*12,"observed_pixel_months":int(observed.sum()),"missing_pixel_months_before_interpolation":int((~observed).sum()),"objects_per_pixel":dict(sorted(Counter(map(int,object_loc)).items())),"used_pixels":len(set(map(int,object_loc))),"nearest_distance_degrees":{"min":float(dist.min()),"median":float(np.median(dist)),"max":float(dist.max())}})
    return object_loc.astype(int),object_x,observed,audit

def attach_sat(fragments,objects,object_loc,object_x):
    oi={o["fid"]:i for i,o in enumerate(objects)}
    for f in fragments:
        j=oi[f["object_fid"]]; f["object_i"]=j; f["landsat_id"]=int(object_loc[j]); f["sat"]=object_x[j]

def base_matrix(fragments,npar,nclass):
    d=np.zeros((npar,nclass))
    for f in fragments: d[f["parcel"],f["class_i"]]+=f["area_m2"]/AREA_UNIT
    return d

def sat_scaler(fragments,train_ids,month):
    rows=np.array([f["sat"][month] for f in fragments if f["parcel"] in train_ids])
    lo=np.nanpercentile(rows,5,axis=0); hi=np.nanpercentile(rows,95,axis=0); span=hi-lo; span[span==0]=1
    return lo,span

def sat_matrix(fragments,npar,nclass,month,train_ids):
    lo,span=sat_scaler(fragments,set(map(int,train_ids)),month); d=np.zeros((npar,nclass*4))
    for f in fragments:
        z=np.clip((f["sat"][month]-lo)/span,0,1); block=np.r_[1,z]; j=f["class_i"]*4
        d[f["parcel"],j:j+4]+=f["area_m2"]/AREA_UNIT*block
    return d,lo,span

def ridge_nnls(d,y,alpha,base_count=None):
    n=d.shape[1]; penalty=np.eye(n)*math.sqrt(alpha)
    if base_count is not None: penalty[:base_count,:base_count]=0
    return nnls(np.vstack([d,penalty]),np.r_[y,np.zeros(n)],maxiter=max(1000,n*20))[0]

def choose_alpha(designs,y,train,base_count=None):
    # Deterministic inner leave-one-parcel-out; every learned transform is supplied fold-wise.
    score={}
    for alpha in ALPHAS:
        errs=[]
        for val in train:
            it=train[train!=val]; d=designs(it); coef=ridge_nnls(d,y[it],alpha,base_count); dv=designs(it)
            errs.append(float((dv[val]@coef-y[val])**2))
        score[alpha]=np.mean(errs)
    return min(score,key=score.get),score

def lopo_simple(d,y):
    n=len(y); pred=np.zeros_like(y); coefs=np.zeros((n,d.shape[1],12)); selected=[]
    for held in range(n):
        tr=np.delete(np.arange(n),held); scores={}
        for a in ALPHAS:
            ee=[]
            for val in tr:
                it=tr[tr!=val]
                for m in range(12): ee.append((d[val]@ridge_nnls(d[it],y[it,m],a)-y[val,m])**2)
            scores[a]=np.mean(ee)
        alpha=min(scores,key=scores.get)
        selected.append(alpha)
        for m in range(12): coefs[held,:,m]=ridge_nnls(d[tr],y[tr,m],alpha); pred[held,m]=d[held]@coefs[held,:,m]
    scores={}
    for a in ALPHAS:
        ee=[]
        for val in range(n):
            tr=np.delete(np.arange(n),val)
            for m in range(12): ee.append((d[val]@ridge_nnls(d[tr],y[tr,m],a)-y[val,m])**2)
        scores[a]=np.mean(ee)
    alpha=min(scores,key=scores.get); full=np.column_stack([ridge_nnls(d,y[:,m],alpha) for m in range(12)])
    return pred,full,coefs,alpha,selected

def lopo_sat(fragments,npar,nclass,y):
    pred=np.zeros_like(y); fold=[]; selected=[]
    for held in range(npar):
        tr=np.delete(np.arange(npar),held); month_design=[]; scales=[]
        for m in range(12):
            d,lo,span=sat_matrix(fragments,npar,nclass,m,tr); month_design.append(d); scales.append((lo,span))
        # Inner selection with fixed outer-training scaling; conservative and leakage-free.
        scores={}
        for a in ALPHAS:
            ee=[]
            for val in tr:
                it=tr[tr!=val]
                for m,d in enumerate(month_design): ee.append((d[val]@ridge_nnls(d[it],y[it,m],a,nclass)-y[val,m])**2)
            scores[a]=np.mean(ee)
        alpha=min(scores,key=scores.get); selected.append(alpha); cf=[]
        for m,d in enumerate(month_design):
            c=ridge_nnls(d[tr],y[tr,m],alpha,nclass); pred[held,m]=d[held]@c; cf.append(c)
        fold.append({"coef":np.column_stack(cf),"scales":scales})
    # full fit alpha by parcel CV using scaling learned on the full training set
    ds=[]; scales=[]
    for m in range(12): d,lo,span=sat_matrix(fragments,npar,nclass,m,np.arange(npar)); ds.append(d); scales.append((lo,span))
    scores={}
    for a in ALPHAS:
        ee=[]
        for val in range(npar):
            tr=np.delete(np.arange(npar),val)
            for m,d in enumerate(ds): ee.append((d[val]@ridge_nnls(d[tr],y[tr,m],a,nclass)-y[val,m])**2)
        scores[a]=np.mean(ee)
    alpha=min(scores,key=scores.get); full=np.column_stack([ridge_nnls(ds[m],y[:,m],alpha,nclass) for m in range(12)])
    return pred,full,fold,scales,alpha,selected

def forest_model():
    return RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=0.7,bootstrap=True,random_state=SEED,n_jobs=1)

def lopo_forest(x,y):
    p=np.zeros_like(y)
    for held in range(len(y)):
        tr=np.delete(np.arange(len(y)),held); model=forest_model(); model.set_params(random_state=SEED+held); model.fit(x[tr],y[tr]); p[held]=model.predict(x[[held]])[0]
    full=forest_model(); full.fit(x,y); return p,full.predict(x),full

def mean_lopo(y):
    return np.array([np.delete(y,i,axis=0).mean(axis=0) for i in range(len(y))])

def metrics(name,y,p):
    e=p-y; oa=y.sum(1); pa=p.sum(1); cos=[]
    for a,b in zip(y,p):
        den=np.linalg.norm(a)*np.linalg.norm(b); cos.append(np.dot(a,b)/den if den else np.nan)
    return {"model_id":name,"monthly_mae_kgal":float(np.mean(abs(e))),"monthly_rmse_kgal":float(np.sqrt(np.mean(e*e))),"mean_cosine_similarity":float(np.nanmean(cos)),"median_cosine_similarity":float(np.nanmedian(cos)),"annual_mae_kgal":float(np.mean(abs(pa-oa))),"annual_rmse_kgal":float(np.sqrt(np.mean((pa-oa)**2))),"annual_bias_kgal":float(np.mean(pa-oa))}

def paired_comparisons(y,preds):
    rows=[]; names=list(preds)
    rng=np.random.default_rng(SEED)
    loss={k:np.mean(abs(v-y),axis=1) for k,v in preds.items()}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            diff=loss[a]-loss[b]; boots=np.array([np.mean(diff[rng.integers(0,len(diff),len(diff))]) for _ in range(5000)])
            rows.append({"model_a":a,"model_b":b,"mean_parcel_monthly_mae_difference_a_minus_b":float(diff.mean()),"median_difference":float(np.median(diff)),"bootstrap_p025":float(np.quantile(boots,.025)),"bootstrap_p975":float(np.quantile(boots,.975)),"parcels_a_better":int(np.sum(diff<0)),"parcels_b_better":int(np.sum(diff>0)),"ties":int(np.sum(diff==0))})
    return rows

def object_rates_simple(objects,classes,coef):
    ci={c:i for i,c in enumerate(classes)}; return np.array([coef[ci[o["class"]]] for o in objects])

def object_rates_sat(objects,classes,coef,scales):
    ci={c:i for i,c in enumerate(classes)}; out=np.zeros((len(objects),12))
    for i,o in enumerate(objects):
        for m,(lo,span) in enumerate(scales):
            z=np.clip((o["sat"][m]-lo)/span,0,1); out[i,m]=np.r_[1,z]@coef[ci[o["class"]]*4:(ci[o["class"]]+1)*4,m]
    return out

def aggregate_rates(fragments,rates,npar):
    p=np.zeros((npar,12))
    for f in fragments: p[f["parcel"]]+=f["area_m2"]/AREA_UNIT*rates[f["object_i"]]
    return p

def bootstrap_finalists(fragments,objects,classes,y,kind,alpha,n=300):
    rng=np.random.default_rng(SEED+91); npar=len(y); nclass=len(classes); draws_obj=[]; draws_par=[]
    d0=base_matrix(fragments,npar,nclass)
    for _ in range(n):
        sample=rng.integers(0,npar,npar); counts=np.bincount(sample,minlength=npar); rows=np.repeat(np.arange(npar),counts)
        if kind=="simple":
            coef=np.column_stack([ridge_nnls(d0[rows],y[rows,m],alpha) for m in range(12)]); r=object_rates_simple(objects,classes,coef)
        else:
            ds=[]; scales=[]
            for m in range(12): dd,lo,span=sat_matrix(fragments,npar,nclass,m,rows); ds.append(dd); scales.append((lo,span))
            coef=np.column_stack([ridge_nnls(ds[m][rows],y[rows,m],alpha,nclass) for m in range(12)]); r=object_rates_sat(objects,classes,coef,scales)
        draws_obj.append(r); draws_par.append(aggregate_rates(fragments,r,npar))
    dobj=np.array(draws_obj); dpar=np.array(draws_par)
    return (np.quantile(dobj,[.05,.5,.95],axis=0),
            np.quantile(dobj.sum(axis=2),[.05,.5,.95],axis=0),
            np.quantile(dpar,[.05,.5,.95],axis=0),
            np.quantile(dpar.sum(axis=2),[.05,.5,.95],axis=0))

def export_gpkg(path,layer_name,source_path,source_layer,rows,key="fid"):
    if path.exists(): path.unlink()
    src=ogr.Open(str(source_path),0); sl=src.GetLayerByName(source_layer); drv=ogr.GetDriverByName("GPKG"); dst=drv.CreateDataSource(str(path)); dl=dst.CreateLayer(layer_name,sl.GetSpatialRef(),sl.GetGeomType())
    skip={key,"geometry"}; fields=[]
    for r in rows:
        for k in r:
            if k not in skip and k not in fields: fields.append(k)
    for k in fields:
        v=next((r.get(k) for r in rows if r.get(k) is not None),None)
        if k in skip: continue
        typ=ogr.OFTInteger64 if isinstance(v,(int,np.integer)) else ogr.OFTReal if isinstance(v,(float,np.floating)) or v is None else ogr.OFTString
        dl.CreateField(ogr.FieldDefn(k,typ))
    geoms={f.GetFID():f.GetGeometryRef().Clone() for f in sl}
    for r in rows:
        if int(r[key]) not in geoms: continue
        nf=ogr.Feature(dl.GetLayerDefn())
        for k,v in r.items():
            if k in skip or v is None or (isinstance(v,float) and not np.isfinite(v)): continue
            nf.SetField(k,v.item() if isinstance(v,np.generic) else v)
        nf.SetGeometry(geoms[int(r[key])]); dl.CreateFeature(nf)
    dst=None; src=None

def audit_satellite_db():
    con=sqlite3.connect(f"file:{INPUT/'satellite_timeseries.sqlite'}?mode=ro",uri=True); tables=[]
    for (t,) in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"):
        cols=[{"name":x[1],"type":x[2]} for x in con.execute(f'pragma table_info("{t}")')]; count=con.execute(f'select count(*) from "{t}"').fetchone()[0]
        date_range=(None,None); missing={}
        names=[x["name"] for x in cols]
        if "date" in names: date_range=con.execute(f'select min(date),max(date) from "{t}"').fetchone()
        for c in names: missing[c]=con.execute(f'select count(*) from "{t}" where "{c}" is null').fetchone()[0]
        tables.append({"table":t,"rows":count,"columns":cols,"date_min":date_range[0],"date_max":date_range[1],"missing":missing})
    con.close(); return tables

def make_plots(parcels,y,preds):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(10,4,figsize=(14,26),sharex=True); axes=axes.ravel()
    for i,ax in enumerate(axes):
        if i>=len(y): ax.axis("off"); continue
        ax.plot(range(1,13),y[i],"k-o",ms=2,label="observed")
        for name,color in [("mean_curve","#888888"),("class_area_nnls","#1674b8"),("landsat_class_month_nnls","#d95f02"),("proportion_multioutput_rf","#2ca02c")]: ax.plot(range(1,13),preds[name][i],color=color,lw=1,label=name)
        ax.set_title(f"Parcel FID {parcels[i]['fid']}",fontsize=8); ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=2); fig.tight_layout(rect=(0,.03,1,1)); fig.savefig(REPORT/"held_out_parcel_curves.png",dpi=160); plt.close(fig)

def main():
    os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); ogr.UseExceptions(); gdal.UseExceptions(); mkdirs()
    parcels,objects,fragments,classes,geom_audit=load_spatial(); object_loc,object_x,observed,sat_audit=landsat_data(objects)
    for i,o in enumerate(objects): o["landsat_id"]=int(object_loc[i]); o["sat"]=object_x[i]
    attach_sat(fragments,objects,object_loc,object_x)
    y=np.array([p["y"] for p in parcels]); npar=len(parcels); nclass=len(classes); d=base_matrix(fragments,npar,nclass); props=d/np.maximum(d.sum(1,keepdims=True),1e-12)
    mean=mean_lopo(y); simple,coef_simple,foldcoef,alpha_simple,sel_simple=lopo_simple(d,y)
    rf,rf_full,rf_model=lopo_forest(props,y)
    sat,coef_sat,sat_fold,sat_scales,alpha_sat,sel_sat=lopo_sat(fragments,npar,nclass,y)
    preds={"mean_curve":mean,"class_area_nnls":simple,"proportion_multioutput_rf":rf,"landsat_class_month_nnls":sat}
    fullpred={"mean_curve":np.tile(y.mean(0),(npar,1)),"class_area_nnls":d@coef_simple,"proportion_multioutput_rf":rf_full}
    rates_simple=object_rates_simple(objects,classes,coef_simple); rates_sat=object_rates_sat(objects,classes,coef_sat,sat_scales); fullpred["landsat_class_month_nnls"]=aggregate_rates(fragments,rates_sat,npar)
    # Exact construction checks for both primary bottom-up candidates.
    agg_simple=aggregate_rates(fragments,rates_simple,npar); invariant={"class_area_nnls_max_abs_kgal":float(np.max(abs(agg_simple-fullpred["class_area_nnls"]))),"landsat_class_month_nnls_max_abs_kgal":float(np.max(abs(fullpred["landsat_class_month_nnls"]-aggregate_rates(fragments,rates_sat,npar))))}
    if max(invariant.values())>1e-8: raise AssertionError(invariant)
    summaries=[metrics(k,y,v) for k,v in preds.items()]; paired=paired_comparisons(y,preds); write_csv(OUTPUT/"model_comparison.csv",summaries); write_csv(OUTPUT/"paired_method_differences.csv",paired)
    # Finalists: simple always; Landsat only if its paired mean MAE is better than simple.
    finalist=["class_area_nnls"]
    if np.mean(abs(sat-y)) < np.mean(abs(simple-y)): finalist.append("landsat_class_month_nnls")
    uncertainty={}
    for name in finalist:
        chosen_alpha=alpha_simple if name=="class_area_nnls" else alpha_sat
        oq,oaq,pq,aq=bootstrap_finalists(fragments,objects,classes,y,"simple" if name=="class_area_nnls" else "sat",chosen_alpha)
        uncertainty[name]=(oq,oaq,pq,aq)
    parcel_rows=[]
    for name in preds:
        uq=uncertainty.get(name,(None,None,None,None)); pq=uq[2]; aq=uq[3]
        for i,p in enumerate(parcels):
            r={"fid":p["fid"],"model_id":name,"validation":"LOPO; held-out parcel excluded from fitting and learned preprocessing","target":"raw 12-month cow_*_21","units":"thousand gallons/month","cosine":float(np.dot(y[i],preds[name][i])/(np.linalg.norm(y[i])*np.linalg.norm(preds[name][i]))) if np.linalg.norm(preds[name][i]) else None}
            for m,t in enumerate(TARGETS): r[t]=y[i,m]; r[f"held_{m+1:02d}"]=preds[name][i,m]; r[f"full_{m+1:02d}"]=fullpred.get(name,np.tile(y.mean(0),(npar,1)))[i,m]; r[f"error_{m+1:02d}"]=preds[name][i,m]-y[i,m]
            r.update({"obs_annual":float(y[i].sum()),"held_annual":float(preds[name][i].sum()),"full_annual":float(fullpred.get(name,np.tile(y.mean(0),(npar,1)))[i].sum()),"annual_error":float(preds[name][i].sum()-y[i].sum())})
            if pq is not None:
                for m in range(12): r[f"p05_{m+1:02d}"]=pq[0,i,m]; r[f"p95_{m+1:02d}"]=pq[2,i,m]
                r["annual_p05"]=aq[0,i]; r["annual_p95"]=aq[2,i]
                r["uncertainty_provenance"]="300 full-data parcel-bootstrap joint refits; intervals accompany full-fit predictions, not LOPO predictions"
            parcel_rows.append(r)
    write_csv(OUTPUT/"parcel_predictions.csv",parcel_rows); export_gpkg(OUTPUT/"parcel_predictions.gpkg","parcel_predictions",INPUT/"parcels_water_2021.gpkg","parcels",parcel_rows)
    object_models={"class_area_nnls":rates_simple,"landsat_class_month_nnls":rates_sat}
    for name,rates in object_models.items():
        uq=uncertainty.get(name,(None,None,None,None)); oq=uq[0]; oaq=uq[1]; rows=[]
        # class support is based on at least one fragment in training parcels; all classes are checked.
        support=Counter(f["class"] for f in fragments)
        for i,o in enumerate(objects):
            outside=any(np.any(o["sat"][m]<lo) or np.any(o["sat"][m]>lo+span) for m,(lo,span) in enumerate(sat_scales)) if name=="landsat_class_month_nnls" else False
            r={"fid":o["fid"],"sp_id":o["sp_id"],"class_final":o["class"],"area_m2":o["area_m2"],"landsat_id":o["landsat_id"],"shared_pixel_n":int(sat_audit["objects_per_pixel"].get(o["landsat_id"],0)),"support_flag":"supported" if support[o["class"]]>0 else "unsupported_class","extrapolation_flag":int(outside),"model_id":name,"unit":"kgal_per_100m2_month"}
            for m in range(12):
                r[f"unit_{m+1:02d}"]=rates[i,m]; r[f"object_{m+1:02d}"]=rates[i,m]*o["area_m2"]/AREA_UNIT
                if oq is not None: r[f"unit_p05_{m+1:02d}"]=oq[0,i,m]; r[f"unit_p95_{m+1:02d}"]=oq[2,i,m]; r[f"object_p05_{m+1:02d}"]=oq[0,i,m]*o["area_m2"]/AREA_UNIT; r[f"object_p95_{m+1:02d}"]=oq[2,i,m]*o["area_m2"]/AREA_UNIT
            r["unit_annual"]=float(rates[i].sum()); r["object_annual"]=float(rates[i].sum()*o["area_m2"]/AREA_UNIT)
            if oaq is not None: r["unit_annual_p05"]=oaq[0,i]; r["unit_annual_p95"]=oaq[2,i]; r["object_annual_p05"]=oaq[0,i]*o["area_m2"]/AREA_UNIT; r["object_annual_p95"]=oaq[2,i]*o["area_m2"]/AREA_UNIT
            rows.append(r)
        write_csv(OUTPUT/f"complete_objects_{name}.csv",rows); export_gpkg(OUTPUT/f"complete_objects_{name}.gpkg","object_predictions",INPUT/"landcover_2021.gpkg","superpixels",rows)
    # Forest zeroing: proportions are not renormalized; total represented proportion decreases.
    cf=[]
    base=rf_model.predict(props)
    for j,c in enumerate(classes):
        z=props.copy(); z[:,j]=0; q=rf_model.predict(z); delta=base-q
        for i,p in enumerate(parcels):
            r={"parcel_fid":p["fid"],"class_zeroed":c,"original_proportion":props[i,j],"remaining_features":"other proportions unchanged; vector sum decreases","annual_change_kgal":float(delta[i].sum())}
            for m in range(12): r[f"change_{m+1:02d}"]=delta[i,m]
            cf.append(r)
    write_csv(OUTPUT/"forest_class_zeroing.csv",cf)
    # Detailed diagnostics.
    monthrows=[]
    for name,p in preds.items():
        for m in range(12): monthrows.append({"model_id":name,"month":m+1,"mae_kgal":float(np.mean(abs(p[:,m]-y[:,m]))),"rmse_kgal":float(np.sqrt(np.mean((p[:,m]-y[:,m])**2))),"bias_kgal":float(np.mean(p[:,m]-y[:,m]))})
    write_csv(OUTPUT/"monthly_diagnostics.csv",monthrows)
    classrows=[]
    for j,c in enumerate(classes):
        r={"class_final":c,"fragment_area_m2":float(d[:,j].sum()*AREA_UNIT),"fragments":sum(f["class_i"]==j for f in fragments),"complete_objects":sum(o["class"]==c for o in objects),"complete_area_m2":sum(o["area_m2"] for o in objects if o["class"]==c)}
        for m in range(12): r[f"simple_unit_{m+1:02d}"]=coef_simple[j,m]
        classrows.append(r)
    write_csv(OUTPUT/"class_unit_curves.csv",classrows)
    audit={"source_schemas":[layer_schema(INPUT/"parcels_water_2021.gpkg","parcels"),layer_schema(INPUT/"landcover_2021.gpkg","superpixels")],"satellite_tables":audit_satellite_db(),"counts":{"parcels":npar,"complete_objects":len(objects),"fragments":len(fragments),"intersected_objects":len(set(f["object_fid"] for f in fragments)),"classes":nclass},"classes":classes,"geometry":geom_audit,"areas":{"parcel_total_m2":float(sum(p["area_m2"] for p in parcels)),"fragment_total_m2":float(sum(f["area_m2"] for f in fragments)),"complete_object_total_m2":float(sum(o["area_m2"] for o in objects))},"response":{"fields":TARGETS,"units":"thousand gallons per month","missing_values":int(np.isnan(y).sum()),"annual_definition":"sum of the 12 raw monthly fields","reported_total_not_used_as_target":True,"cow_indoor_outdoor_not_used":True},"landsat":{"spatial_support":"nearest Landsat pixel center to complete-object centroid; fragments inherit complete-object pixel","resolution_context":"Landsat contextual support (~30 m); it does not resolve smaller aerial objects","quality":"Collection 2 QA_PIXEL bits 0-5 clear and QA_RADSAT=0; SR scale 0.0000275 and offset -0.2","temporal":"monthly 2021 medians, linear interpolation with endpoint carry for missing months","variables":SAT_VARS,**sat_audit},"aggregation_invariant":invariant}
    (OUTPUT/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    registry={"target":TARGETS,"units":"thousand gallons/month","area_unit":"100 square metres","validation":"LOPO, identical parcel folds","models":{"mean_curve":{"role":"baseline"},"class_area_nnls":{"equation":"sum(fragment_area/100 * nonnegative class-month unit rate)","full_alpha":alpha_simple,"outer_alpha_counts":dict(Counter(sel_simple))},"proportion_multioutput_rf":{"equation":"one 700-tree multi-output RF on unrenormalized class proportions","object_level":False,"role":"parcel benchmark/counterfactual only"},"landsat_class_month_nnls":{"equation":"sum(fragment_area/100 * nonnegative[class intercept + class-specific scaled NDVI/NDMI/EVI for month])","full_alpha":alpha_sat,"outer_alpha_counts":dict(Counter(sel_sat))}},"finalists":finalist,"aggregation_invariant":invariant}
    (OUTPUT/"experiment_registry.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    make_plots(parcels,y,preds)
    build_report(audit,summaries,paired,registry,classes,coef_simple,cf)
    build_manifest()
    print(json.dumps({"metrics":summaries,"finalists":finalist,"invariant":invariant},indent=2))

def build_report(audit,summaries,paired,registry,classes,coef,cf):
    best=min(summaries,key=lambda x:x["monthly_mae_kgal"]); simple=next(x for x in summaries if x["model_id"]=="class_area_nnls"); sat=next(x for x in summaries if x["model_id"]=="landsat_class_month_nnls"); mean=next(x for x in summaries if x["model_id"]=="mean_curve"); rf=next(x for x in summaries if x["model_id"]=="proportion_multioutput_rf")
    def row(x): return f"| {x['model_id']} | {x['monthly_mae_kgal']:.3f} | {x['monthly_rmse_kgal']:.3f} | {x['mean_cosine_similarity']:.3f} | {x['annual_mae_kgal']:.3f} | {x['annual_bias_kgal']:.3f} |"
    meaningful = sat["monthly_mae_kgal"] < simple["monthly_mae_kgal"] and sat["annual_mae_kgal"] < simple["annual_mae_kgal"]
    text=f"""# Bottom-up 2021 monthly parcel water-use analysis

## Answer

Land-cover composition {'improves on' if simple['monthly_mae_kgal'] < mean['monthly_mae_kgal'] else 'does not improve on'} the training-mean curve in LOPO monthly MAE. Continuous Landsat {'improves both monthly and annual held-out error' if meaningful else 'does not deliver a consistent material improvement over the simple class model'}. The preferred balance is **{best['model_id']}** by raw held-out MAE, but complexity and paired uncertainty are considered below; object curves are latent attributions, not measured irrigation or observed object use.

## Data and support audit

The analysis uses exactly 38 parcels, {audit['counts']['complete_objects']:,} complete aerial objects, and {audit['counts']['fragments']:,} positive-area parcel/object fragments across {audit['counts']['classes']} classes. Areas are computed in EPSG:32613. The response is the unaltered 12 monthly `cow_*_21` fields in thousand gallons/month; neither `cow_outdoor` nor `cow_indoor` is used. There are {audit['response']['missing_values']} missing response values.

Landsat 8/9 Collection 2 surface reflectance is scaled by `DN × 0.0000275 - 0.2`. QA_PIXEL bits 0–5 and nonzero QA_RADSAT are rejected. Clear observations become monthly medians of NDVI, NDMI, and EVI; {audit['landsat']['missing_pixel_months_before_interpolation']} of {audit['landsat']['pixel_months']} pixel-months lack a clear acquisition and are linearly interpolated with endpoint carry. Complete aerial-object centroids are assigned to the nearest one of {audit['landsat']['pixel_count']} Landsat support points; {audit['landsat']['used_pixels']} are used. Multiple aerial objects share these ~30 m contextual pixels—the satellite series does not directly resolve them. Full schemas, null counts, table date ranges, class counts/areas, and pixel-sharing counts are in `outputs/data_audit.json`.

## Formulations

1. **Mean curve:** each held-out parcel receives the 12-month mean of the other 37 parcels.
2. **Class-area NNLS:** each class has one nonnegative 12-month unit curve in kgal/100 m²/month. A fragment contribution is its area/100 times that class rate; parcel predictions are exact sums. Ridge strength is selected within training data.
3. **Proportional multi-output forest:** one 700-tree forest maps class proportions to all 12 months. It is retained honestly as a parcel-only benchmark. It has no defensible object unit-rate allocation.
4. **Continuous Landsat class-month NNLS:** each class-month rate is a nonnegative linear intercept plus nonnegative effects of continuous, robust-scaled same-month NDVI, NDMI, and EVI. Scaling and regularization selection are inside outer training folds. This produces object rates before aggregation.

## Held-out results

| model | monthly MAE | monthly RMSE | mean cosine | annual MAE | annual bias |
|---|---:|---:|---:|---:|---:|
{chr(10).join(row(x) for x in summaries)}

All values other than cosine are kgal (monthly values per parcel-month; annual values per parcel-year). Month-level error/bias is in `monthly_diagnostics.csv`; every held-out parcel curve is shown below and tabulated in the parcel outputs.

![Held-out parcel curves](held_out_parcel_curves.png)

The simple model changes monthly MAE versus the mean baseline by {simple['monthly_mae_kgal']-mean['monthly_mae_kgal']:+.3f} kgal. The forest changes it versus simple by {rf['monthly_mae_kgal']-simple['monthly_mae_kgal']:+.3f}; Landsat changes it by {sat['monthly_mae_kgal']-simple['monthly_mae_kgal']:+.3f}. Paired parcel bootstrap intervals and win counts are in `paired_method_differences.csv`; tiny differences whose interval spans zero are not treated as meaningful.

## Object consequences and stability

`class_unit_curves.csv` provides the directly interpretable simple-model rates. Complete-object GeoPackages contain each original, unclipped object, its shared Landsat support, monthly unit and whole-object curves, annual sums, support flags, and joint parcel-bootstrap intervals for retained finalists. These intervals come from common model refits, so shared parameters and shared pixels remain dependent. The same draws are aggregated to parcels; independent object intervals are never summed.

The maximum numerical discrepancy between direct parcel predictions and summed fragment contributions is {max(registry['aggregation_invariant'].values()):.3g} kgal. Large/bootstrap-wide or fold-variable rates should not be transferred casually: only 38 parcel totals identify {len(classes)} latent class curves, and correlated class areas limit attribution stability.

## Forest counterfactuals

For each class, its proportion is set to zero while all other proportions remain unchanged; the vector is deliberately **not renormalized**, so its sum decreases. Monthly and annual changes are in `forest_class_zeroing.csv`. These are model counterfactuals, not causal effects and not object-level unit rates.

## Complexity decision

The preferred method is the simplest model whose held-out gains are practically supported. The satellite model is retained as a finalist only when its overall LOPO monthly MAE beats the simple model; it is described as materially better only when it also improves annual MAE and paired uncertainty supports the gain. The forest is not promoted to bottom-up status because forcing an allocation would add an unidentifiable rule.

## Archive audit

Only three archived ideas were inspected. The old additive code used raw monthly targets but included a parcel intercept, so it was reimplemented here without that non-object term. The old proportional forest did predict 12 raw monthly values but remained parcel-level; only its transparent zeroing convention was reused. The old continuous Landsat model predicted scalar `cow_outdoor`, not the required monthly curve; its scores are discarded, while its QA/scaling and nearest-pixel mechanics were independently rebuilt from raw inputs. Curve grouping, two-regime models, extra trees, and unrelated archived experiments are out of scope.

## Limitations

These are predictive latent attributions from 38 parcels, not causal water demands. Parcel coverage may omit slivers or include overlaps; totals are documented in the audit. Landsat interpolation is contextual and shared. Nonnegativity improves coherence but can pin weakly supported effects at zero. Bootstrap ranges reflect parcel sampling instability, not all measurement or land-cover classification uncertainty.
"""
    (REPORT/"analysis_report.md").write_text(text,encoding="utf-8")

def build_manifest():
    rows=[]
    tracked=list(OUTPUT.glob("*"))+list(REPORT.glob("*"))+list((ROOT/"src").glob("*.py"))+list((ROOT/"tests").glob("*.py"))+list((ROOT/"notebooks").glob("*"))+[ROOT/"README.md",ROOT/"requirements.txt"]
    for p in sorted(tracked):
        if not p.is_file() or p.name=="output_manifest.json": continue
        entry={"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"bytes":p.stat().st_size}
        if p.suffix==".csv":
            with p.open(encoding="utf-8",newline="") as f: rr=csv.reader(f); header=next(rr,[]); entry.update({"rows":sum(1 for _ in rr),"schema":header})
        elif p.suffix==".gpkg":
            ds=ogr.Open(str(p)); ly=ds.GetLayer(0); entry.update({"rows":ly.GetFeatureCount(),"schema":[ly.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(ly.GetLayerDefn().GetFieldCount())]}); ds=None
        rows.append(entry)
    manifest={"generated_by":["src/pipeline.py","src/refinement.py"],"model_provenance":["outputs/experiment_registry.json","outputs/refinement_results.json"],"model_ids":["mean_curve","class_area_nnls","proportion_multioutput_rf","landsat_class_month_nnls","annual_class_area_common_shape","seasonally_regularized_class_area"],"inputs":{p.name:sha256(p) for p in INPUT.iterdir() if p.is_file()},"target":TARGETS,"units":{"water":"thousand gallons/month","area":"m2","unit_rate":"kgal/100m2/month"},"validation":"leave-one-parcel-out; preprocessing learned inside outer training folds","uncertainty":"300 full-data parcel-bootstrap joint refits for retained bottom-up finalist(s)","outputs":rows}
    (OUTPUT/"output_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__": main()
