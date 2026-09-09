#!/usr/bin/env python3
"""Size-class extension using verified v1 features, targets and outer folds."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pyogrio.raw
from threadpoolctl import threadpool_limits
import analyze_seasonal as core

ROOT=Path(__file__).resolve().parents[1]


def assign_bins(area,breaks):
    return np.searchsorted(np.asarray(breaks),area,side='right')


def fit_fold(cfg,features,parcel,support,area,y,folds,held):
    tr=np.flatnonzero(folds!=held); va=np.flatnonzero(folds==held)
    if len(tr)<cfg['inner_folds'] or len(va)==0:
        raise ValueError('Insufficient class-specific fold population')
    alpha,errors=core.choose_alpha(cfg,features,parcel,support,area,y,tr)
    d,lo,span=core.design(features,parcel,support,area,tr,len(y))
    coef=core.fit(d,y,tr,alpha)
    return va,core.predict(d,coef)[va],y[tr].mean(0),coef,lo,span,alpha,errors


def comparison(y,p):
    result=core.scores(y,p)
    total=y.sum(1); predicted=p.sum(1); denom=total+predicted
    departure=np.divide(total-predicted,denom,out=np.full(len(y),np.nan),where=denom>0)
    result.update(underpredicted_fraction=float(np.mean(predicted<total)),
                  observed_total_kgal=float(total.sum()),predicted_total_kgal=float(predicted.sum()),
                  mean_relative_departure=float(np.nanmean(departure)),
                  median_relative_departure=float(np.nanmedian(departure)))
    return result


def reconstruct(features,spatial,state,frame):
    pred=np.full((len(frame),12),np.nan)
    row=spatial['parcel']; support=spatial['support']; area=spatial['area']
    for cls in sorted(frame.loc[frame.status=='eligible','size_class_id'].unique()):
        for fold in range(1,6):
            held=(frame.size_class_id.to_numpy()==cls)&(frame.fold.to_numpy()==fold)
            fragments=held[row]; key=f'class_{cls}_fold_{fold}'
            z=np.clip((features-state[key+'_lo'])/state[key+'_span'],0,1)
            coef=state[key+'_coef']; accum=np.zeros((len(frame),12))
            for m in range(12):
                rate=coef[m,0]+z[support[fragments],m]@coef[m,1:]
                np.add.at(accum[:,m],row[fragments],area[fragments]/100*rate)
            pred[held]=accum[held]
    return pred


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'analysis/size_binned/config.json');args=ap.parse_args()
    cfg=json.loads(args.config.read_text());src=ROOT/cfg['source_run'];out=ROOT/cfg['output_directory'];out.mkdir(parents=True,exist_ok=True)
    cache_names=['parcel_predictions.csv','spatial.npz','satellite_monthly.npz','fold_assignments.csv','Westminster_2021_observed_predicted.gpkg']
    checks={r['path']:r['sha256'] for r in json.loads((src/'artifact_checksums.json').read_text())}
    inputs=[]
    for name in cache_names:
        path=src/name;digest=core.sha256(path)
        if checks[str(path.relative_to(ROOT))]!=digest: raise ValueError(f'Original artifact changed: {name}')
        inputs.append(dict(path=str(path),sha256=digest))
    for path in [Path(__file__),ROOT/'scripts/analyze_seasonal.py',args.config,ROOT/'analysis/size_binned/requirements.lock.txt',ROOT/'analysis/size_binned/METHODS.md',src/'manifest.json',src/'artifact_checksums.json']:
        inputs.append(dict(path=str(path.resolve()),sha256=core.sha256(path)))
    manifest=dict(inputs=inputs,config=cfg,python=sys.version)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:
        raise ValueError('Manifest changed; select a new output directory')
    core.dump(out/'manifest.json',manifest)
    old=pd.read_csv(src/'parcel_predictions.csv',dtype={'parcel_apn':str})
    spatial=np.load(src/'spatial.npz');sat=np.load(src/'satellite_monthly.npz');features=sat['features']
    assert np.array_equal(spatial['support_id'],sat['support_id'])
    ids=old.parcel_id.to_numpy();fold_assignments=pd.read_csv(src/'fold_assignments.csv')
    assert np.array_equal(ids,fold_assignments.parcel_id)
    assert np.array_equal(old.fold,fold_assignments.fold)
    meta,_,wkb,fields=pyogrio.raw.read(src/'Westminster_2021_observed_predicted.gpkg',layer='parcel_water_use',columns=['parcel_id'])
    geom_by_id=dict(zip(fields[0],wkb));wkb=np.array([geom_by_id[i] for i in ids],dtype=object)
    bins=assign_bins(old.parcel_area_m2.to_numpy(),cfg['area_breaks_m2'])
    eligible=old.status.eq('eligible').to_numpy()
    all_y=old[[f'observed_{m:02d}' for m in range(1,13)]].to_numpy(float)
    original=old[[f'predicted_{m:02d}' for m in range(1,13)]].to_numpy(float)
    predictions=np.full_like(all_y,np.nan);baseline=np.full_like(all_y,np.nan)
    states={};selections=[];groups=[];training_rows=[]
    for cls,label in enumerate(cfg['class_labels']):
        select=np.flatnonzero(eligible&(bins==cls));n=len(select)
        mapping=np.full(len(old),-1,dtype=int);mapping[select]=np.arange(n)
        take=mapping[spatial['parcel']]>=0
        parcel=mapping[spatial['parcel'][take]];support=spatial['support'][take];area=spatial['area'][take]
        y=all_y[select];folds=old.fold.to_numpy()[select]
        core.log(f'Class {cls} {label}: {n:,} eligible parcels')
        for fold in range(1,6):
            va,p,mean,coef,lo,span,alpha,errors=fit_fold(cfg,features,parcel,support,area,y,folds,fold)
            predictions[select[va]]=p;baseline[select[va]]=mean
            key=f'class_{cls}_fold_{fold}'
            states.update({key+'_coef':coef,key+'_lo':lo,key+'_span':span})
            selections.append(dict(size_class_id=cls,fold=fold,alpha=alpha,inner_mse_by_alpha=errors))
            training_rows.append(dict(size_class_id=cls,fold=fold,training_parcels=int(np.sum(folds!=fold)),held_out_parcels=len(va)))
        train=np.arange(n);alpha,errors=core.choose_alpha(cfg,features,parcel,support,area,y,train)
        d,lo,span=core.design(features,parcel,support,area,train,n);coef=core.fit(d,y,train,alpha)
        key=f'class_{cls}_full';states.update({key+'_coef':coef,key+'_lo':lo,key+'_span':span})
        selections.append(dict(size_class_id=cls,fold='full',alpha=alpha,inner_mse_by_alpha=errors))
        groups.append(dict(size_class_id=cls,size_class=label,parcels=n,
                           binned=comparison(y,predictions[select]),original=comparison(y,original[select]),
                           class_mean_reference=comparison(y,baseline[select])))
        core.log(f'Class {label} finished: underpredicted {groups[-1]["binned"]["underpredicted_fraction"]:.1%}')
    assert np.isfinite(predictions[eligible]).all()
    drop=[c for c in old if c.startswith('predicted_') or c.startswith('delta_') or c in ['sum_abs_delta','observed_total','annual_delta']]
    frame=core.add_output_fields(old.drop(columns=drop),predictions)
    frame['size_class_id']=bins
    frame['size_class']=np.array(cfg['class_labels'],dtype=object)[bins]
    frame['signed_departure']=frame.observed_total-frame.predicted_total
    denom=frame.observed_total+frame.predicted_total
    frame['relative_departure']=np.divide(frame.signed_departure,denom,out=np.full(len(frame),np.nan),where=denom>0)
    frame.to_csv(out/'parcel_predictions.csv',index=False)
    frame[['parcel_id','status','fold','size_class_id','size_class']].to_csv(out/'fold_assignments.csv',index=False)
    pd.DataFrame(training_rows).to_csv(out/'training_counts.csv',index=False)
    pd.DataFrame(baseline[eligible],index=ids[eligible],columns=[f'predicted_{m:02d}' for m in range(1,13)]).rename_axis('parcel_id').to_csv(out/'class_mean_reference.csv')
    np.savez_compressed(out/'model_state.npz',**states);core.dump(out/'model_selection.json',selections)
    metrics=dict(eligible_parcels=int(eligible.sum()),retained_parcels=len(frame),
                 binned=comparison(all_y[eligible],predictions[eligible]),
                 original=comparison(all_y[eligible],original[eligible]),
                 class_mean_reference=comparison(all_y[eligible],baseline[eligible]),by_size_class=groups)
    core.dump(out/'metrics.json',metrics)
    path=core.write_gpkg(out,frame,wkb,meta['crs'],cfg)
    final=out/'Westminster_2021_size_binned.gpkg';path.replace(final)
    con=sqlite3.connect(final)
    con.executemany('INSERT INTO analysis_metadata VALUES (?,?)',[
        ('size_class_definition','Area m², lower-inclusive/upper-exclusive: <500; 500–1000; 1000–2000; 2000–10000; 10000–50000; >=50000'),
        ('signed_departure_definition','observed_total minus predicted_total; kgal; positive means observed exceeds expectation'),
        ('relative_departure_definition','(observed_total-predicted_total)/(observed_total+predicted_total); NULL for zero denominator or missing values')])
    con.commit();assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok';assert con.execute('PRAGMA foreign_key_check').fetchall()==[]
    saved=pd.read_sql_query('SELECT * FROM parcel_water_use ORDER BY parcel_id',con);con.close()
    assert np.array_equal(saved.parcel_id,ids)
    for prefix in ['observed','predicted','delta']:
        cols=[f'{prefix}_{m:02d}' for m in range(1,13)]
        np.testing.assert_allclose(saved[cols],frame[cols],equal_nan=True,rtol=0,atol=0)
    np.testing.assert_allclose(saved.sum_abs_delta,np.sum(abs(predictions-all_y),axis=1),equal_nan=True)
    np.testing.assert_allclose(saved.signed_departure,np.sum(all_y-predictions,axis=1),equal_nan=True)
    np.testing.assert_allclose(saved.relative_departure,frame.relative_departure,equal_nan=True)
    _,_,output_wkb,output_fields=pyogrio.raw.read(final,layer='parcel_water_use',columns=['parcel_id'])
    assert all(g==geom_by_id[i] for i,g in zip(output_fields[0],output_wkb))
    rebuilt=reconstruct(features,spatial,np.load(out/'model_state.npz'),frame)
    np.testing.assert_allclose(rebuilt,predictions,equal_nan=True,rtol=1e-12,atol=1e-8)
    core.dump(out/'validation.json',dict(integrity_check='ok',foreign_key_check='ok',
        eligible_parcels=int(eligible.sum()),original_observations_geometry_and_folds_preserved=True,
        saved_monthly_predictions_and_departures_verified=True,
        reconstruction_max_abs_kgal=float(np.nanmax(abs(rebuilt-predictions))),geopackage_sha256=core.sha256(final)))
    core.log(f'Finished and validated: {final}')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
