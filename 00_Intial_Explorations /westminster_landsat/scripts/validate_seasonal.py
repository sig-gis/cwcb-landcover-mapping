#!/usr/bin/env python3
"""Independent checks of the saved spatial deliverable and fold model states."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pyogrio.raw
import shapely

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/seasonal_2021'
SOURCE = ROOT/'prior analysis/add_par_all/Parcels_CoW_2021.gpkg'
GPKG = OUT/'Westminster_2021_observed_predicted.gpkg'
MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()


def main():
    c=sqlite3.connect(GPKG.resolve().as_uri()+'?mode=ro',uri=True)
    assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    df=pd.read_sql_query('SELECT * FROM parcel_water_use ORDER BY parcel_id',c)
    assert len(df)==30912 and df.parcel_id.is_unique
    c.close()
    src=sqlite3.connect(SOURCE.resolve().as_uri()+'?mode=ro',uri=True)
    accounts=pd.read_sql_query('SELECT parcel_fid,'+','.join('cow_'+m+'_21' for m in MONTHS)+' FROM cow_accounts_joined',src)
    original=accounts.groupby('parcel_fid').agg(lambda x: x.sum() if x.notna().all() else np.nan)
    actual=df[[f'observed_{m:02d}' for m in range(1,13)]].to_numpy(float)
    np.testing.assert_array_equal(actual,original.loc[df.parcel_id].to_numpy(float))
    predicted=df[[f'predicted_{m:02d}' for m in range(1,13)]].to_numpy(float)
    delta=df[[f'delta_{m:02d}' for m in range(1,13)]].to_numpy(float)
    np.testing.assert_allclose(delta,predicted-actual,rtol=0,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(df.sum_abs_delta,abs(delta).sum(axis=1),rtol=0,atol=1e-9,equal_nan=True)
    np.testing.assert_allclose(df.annual_delta,delta.sum(axis=1),rtol=0,atol=1e-9,equal_nan=True)
    eligible=df.status.eq('eligible').to_numpy()
    assert eligible.sum()==30274
    assert np.isfinite(predicted[eligible]).all() and (predicted[eligible]>=0).all()
    assert np.isnan(predicted[~eligible]).all()
    assert df.loc[eligible,'fold'].isin([1,2,3,4,5]).all()
    assert df.loc[~eligible,'fold'].eq(-1).all()
    folds=pd.read_csv(OUT/'fold_assignments.csv').set_index('parcel_id')
    np.testing.assert_array_equal(df.fold,folds.loc[df.parcel_id,'fold'])

    # Check source identifiers against named source fields, independent of writer order.
    ids=pd.read_sql_query('SELECT fid,OBJECTID,APN FROM parcels_cow_summary',src).set_index('fid')
    np.testing.assert_array_equal(df.parcel_objectid,ids.loc[df.parcel_id,'OBJECTID'])
    np.testing.assert_array_equal(df.parcel_apn.fillna(''),ids.loc[df.parcel_id,'APN'].fillna(''))
    src.close()
    sm,sids,sg,_=pyogrio.raw.read(SOURCE,layer='parcels_cow_summary',columns=[],return_fids=True)
    om,_,og,fields=pyogrio.raw.read(GPKG,layer='parcel_water_use',columns=['parcel_id'])
    assert sm['crs']==om['crs']
    source_geometry={int(i):g for i,g in zip(sids,sg)}
    geometry_matches=0
    for ident,wkb in zip(fields[0],og):
        source=shapely.from_wkb(source_geometry[int(ident)])
        if source.geom_type=='Polygon': source=shapely.MultiPolygon([source])
        assert shapely.equals_exact(source,shapely.from_wkb(wkb),tolerance=0)
        geometry_matches+=1

    # Reconstruct all held-out predictions from persisted scalers, coefficients,
    # support features and parcel/pixel areas (no model fitting here).
    spatial=np.load(OUT/'spatial.npz')
    satellite=np.load(OUT/'satellite_monthly.npz')
    state=np.load(OUT/'model_state.npz')
    assert np.array_equal(spatial['support_id'],satellite['support_id'])
    row=spatial['parcel']; support=spatial['support']; area=spatial['area']
    largest=0.
    for fold in range(1,6):
        held=df.fold.to_numpy()==fold
        fragments=held[row]
        z=np.clip((satellite['features']-state[f'fold_{fold}_lo'])/state[f'fold_{fold}_span'],0,1)
        coef=state[f'fold_{fold}_coef']
        reconstructed=np.zeros((len(df),12))
        for m in range(12):
            rates=coef[m,0]+z[support[fragments],m]@coef[m,1:]
            np.add.at(reconstructed[:,m],row[fragments],area[fragments]/100*rates)
        error=float(np.max(abs(reconstructed[held]-predicted[held])))
        largest=max(largest,error)
        np.testing.assert_allclose(reconstructed[held],predicted[held],rtol=1e-12,atol=1e-9)
    result={'integrity_check':'ok','foreign_key_check':'ok',
            'parcel_count':len(df),'held_out_prediction_count':int(eligible.sum()),
            'source_monthly_observations':'all matched, including NULLs',
            'geometry_and_identifiers':'all matched source; polygon promotion only',
            'geometry_count':geometry_matches,'delta_and_absolute_sum':'all verified',
            'held_out_model_reconstruction_max_abs_kgal':largest,
            'geopackage_sha256':hashlib.sha256(GPKG.read_bytes()).hexdigest()}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    # Freeze the validated artifacts required by the subsequent size-class fit.
    names = ['parcel_predictions.csv', 'spatial.npz', 'satellite_monthly.npz',
             'fold_assignments.csv', 'Westminster_2021_observed_predicted.gpkg']
    checks = []
    for name in names:
        path = OUT/name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        checks.append(dict(path=str(path.relative_to(ROOT)),
                           bytes=path.stat().st_size, sha256=digest))
    (OUT/'artifact_checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
