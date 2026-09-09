#!/usr/bin/env python3
"""Reproducible expanded-data version of the retained Landsat-only NNLS model.

See analysis/METHODS.md. Original equations and solvers are archived in
analysis/original_src. No indoor/outdoor consumption field is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import pyogrio.raw
from pyproj import Transformer
from scipy.optimize import nnls
from sklearn.model_selection import KFold
import shapely
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def observations(conn):
    """Never convert an absent account/month observation to a measured zero."""
    cols = [f'cow_{m}_21' for m in MONTHS]
    monthly = ','.join(f'CASE WHEN count({c})=count(*) THEN sum({c}) END AS observed_{i+1:02d}'
                       for i, c in enumerate(cols))
    q = f'''SELECT parcel_fid AS parcel_id, count(*) AS account_count,
        sum(join_method='exact_unique_parcel_address_point_conflict') AS conflict_accounts,
        {monthly} FROM cow_accounts_joined GROUP BY parcel_fid ORDER BY parcel_fid'''
    return pd.read_sql_query(q, conn)


def prepare_spatial(cfg, out):
    pc = readonly(ROOT / cfg['parcel_database'])
    obs = observations(pc)
    pc.close()
    meta, fids, wkb, columns = pyogrio.raw.read(
        ROOT / cfg['parcel_database'], layer='parcels_cow_summary',
        columns=['OBJECTID', 'APN'], return_fids=True)
    index = {int(fid): i for i, fid in enumerate(fids)}
    take = np.array([index[int(fid)] for fid in obs.parcel_id])
    wkb = wkb[take]
    obs['parcel_objectid'] = columns[0][take]
    obs['parcel_apn'] = columns[1][take]
    geom = shapely.from_wkb(wkb)
    transformer = Transformer.from_crs(meta['crs'], cfg['area_crs'], always_xy=True)
    projected = shapely.transform(geom, transformer.transform, interleaved=False)
    valid = shapely.is_valid(projected) & ~shapely.is_empty(projected)
    area = shapely.area(projected)
    obs['parcel_area_m2'] = area
    obs['geometry_valid'] = valid.astype(int)

    con = readonly(ROOT / cfg['satellite_database'])
    locations = np.array(con.execute('SELECT id,lon,lat FROM landsat_8_unique_locs ORDER BY id').fetchall())
    con.close()
    x, y = transformer.transform(locations[:, 1], locations[:, 2])
    # These exports form a UTM 30 m grid. Fail rather than assume arbitrary supports
    # are cells on that grid. Tolerance allows CRS floating-point roundoff only.
    step = cfg['grid_size_m']
    ox, oy = x[0] % step, y[0] % step
    if not (np.allclose((x-ox)/step, np.round((x-ox)/step), atol=1e-5, rtol=0)
            and np.allclose((y-oy)/step, np.round((y-oy)/step), atol=1e-5, rtol=0)):
        raise ValueError('Satellite supports are not the configured regular UTM grid')
    xy = np.column_stack([np.round((x-ox)/step), np.round((y-oy)/step)])
    _, first = np.unique(xy, axis=0, return_index=True)
    first = np.sort(first)
    log(f'Grid: consolidating {len(locations)-len(first)} coordinate-roundoff aliases')
    locations, x, y = locations[first], x[first], y[first]
    cells = shapely.box(x-step/2, y-step/2, x+step/2, y+step/2)
    tree = shapely.STRtree(cells)
    ps, ss, aa = [], [], []
    for begin in range(0, len(obs), 2000):
        end = min(begin+2000, len(obs))
        select = np.arange(begin, end)[valid[begin:end]]
        pair = tree.query(projected[select], predicate='intersects')
        parcel = select[pair[0]]
        support = pair[1]
        a = shapely.area(shapely.intersection(projected[parcel], cells[support]))
        keep = a > 1e-8
        ps.extend(parcel[keep]); ss.extend(support[keep]); aa.extend(a[keep])
    ps, ss, aa = np.array(ps, dtype=int), np.array(ss, dtype=int), np.array(aa)
    covered = np.bincount(ps, weights=aa, minlength=len(obs))
    obs['coverage_fraction'] = np.divide(covered, area, out=np.zeros(len(area)), where=area>0)
    support_ids = locations[:, 0].astype(np.int64)
    used = np.unique(ss)
    mapping = np.full(len(locations), -1, dtype=int)
    mapping[used] = np.arange(len(used))
    ss = mapping[ss]
    pd.DataFrame({'parcel_id': obs.parcel_id.to_numpy()[ps], 'support_id': support_ids[used][ss],
                  'intersection_area_m2': aa}).to_csv(out/'parcel_supports.csv', index=False)
    obs.to_csv(out/'observations_and_coverage.csv', index=False)
    np.savez_compressed(out/'spatial.npz', parcel=ps, support=ss, area=aa,
                        support_id=support_ids[used], lon=locations[used, 1], lat=locations[used, 2])
    log(f'Spatial: {len(obs):,} matched parcels, {len(ps):,} intersections, {len(used):,} supports')
    return obs, wkb, meta['crs'], ps, ss, aa, support_ids[used]


def reduce_observations(rows):
    """Original C2 reflectance, QA, range checks, and per-month index medians."""
    columns = ['SR_B2','SR_B4','SR_B5','SR_B6','QA_PIXEL','QA_RADSAT','date','unique_loc_id','image_id']
    df = pl.DataFrame(rows, schema=columns, orient='row', infer_schema_length=None)
    raw = len(df)
    df = df.unique(maintain_order=True)
    unique = len(df)
    df = df.drop_nulls(['SR_B2','SR_B4','SR_B5','SR_B6','QA_PIXEL'])
    present = len(df)
    df = df.filter((pl.col('QA_PIXEL').cast(pl.Int64) & 63) == 0)
    qa = len(df)
    df = df.filter(pl.col('QA_RADSAT').is_null() | (pl.col('QA_RADSAT') == 0))
    saturation = len(df)
    df = df.with_columns([(pl.col(c).cast(pl.Float64)*0.0000275-0.2).alias(c)
                          for c in columns[:4]])
    b,r,n,s = [pl.col(c) for c in columns[:4]]
    df = df.with_columns(((n-r)/(n+r)).alias('ndvi'), ((n-s)/(n+s)).alias('ndmi'),
                         (2.5*(n-r)/(n+6*r-7.5*b+1)).alias('evi'))
    df = df.filter(pl.all_horizontal([pl.col(c).is_finite() for c in ['ndvi','ndmi','evi']]) &
                   (pl.col('ndvi').abs() <= 1) & (pl.col('ndmi').abs() <= 1))
    valid = len(df)
    df = df.with_columns(pl.col('date').str.slice(5,2).cast(pl.Int64).alias('month'))
    result = df.group_by('unique_loc_id','month').agg(
        pl.col('ndvi').median(), pl.col('ndmi').median(), pl.col('evi').median(), pl.len().alias('clear_count'))
    return result, dict(raw=raw, duplicate_alias_observations=raw-unique,
                        missing_required_bands=unique-present, masked_qa=present-qa,
                        masked_saturation=qa-saturation, invalid_index=saturation-valid, clear=valid)


def satellite(cfg, out, support_ids):
    cache = out/'satellite_monthly.npz'
    if cache.exists():
        z = np.load(cache)
        if not np.array_equal(z['support_id'], support_ids):
            raise ValueError('Cached satellite supports changed')
        log('Reusing verified satellite feature cache')
        return z['features'], z['clear_count']
    con = readonly(ROOT / cfg['satellite_database'])
    con.execute('PRAGMA cache_size=-131072')
    # Sensor IDs are local to each table. Join by coordinates, never by an ID
    # assumed to mean the same location for both sensors.
    transform = Transformer.from_crs(4326,cfg['area_crs'],always_xy=True)
    locations = {}
    for sensor in [8,9]:
        values = np.array(con.execute(f'SELECT id,lon,lat FROM landsat_{sensor}_unique_locs ORDER BY id').fetchall())
        x,y = transform.transform(values[:,1],values[:,2])
        locations[sensor] = (values[:,0].astype(int),np.round(np.c_[x,y]/cfg['grid_size_m']).astype(int))
    keys8 = {int(i):tuple(k) for i,k in zip(*locations[8])}
    canonical = {keys8[int(i)]:int(i) for i in support_ids}
    by_sensor = {}
    aliases = []
    for sensor in [8,9]:
        by_sensor[sensor] = {}
        for ident,key in zip(*locations[sensor]):
            if tuple(key) in canonical:
                target = canonical[tuple(key)]
                by_sensor[sensor].setdefault(target,[]).append(int(ident))
                aliases.append((sensor,int(ident),target))
    pd.DataFrame(aliases,columns=['sensor','source_location_id','canonical_support_id']).to_csv(out/'support_aliases.csv',index=False)
    features = np.full((len(support_ids),12,3), np.nan)
    counts = np.zeros((len(support_ids),12), dtype=int)
    audit = {}
    size = cfg['sql_support_batch_size']
    for begin in range(0, len(support_ids), size):
        ids = support_ids[begin:begin+size]
        rows = []
        for sensor in [8,9]:
            mapping = {source:int(i) for i in ids for source in by_sensor[sensor].get(int(i),[])}
            if not mapping:
                continue
            placeholders = ','.join('?' for _ in mapping)
            q = f'''SELECT SR_B2,SR_B4,SR_B5,SR_B6,QA_PIXEL,QA_RADSAT,date,unique_loc_id,image_id
                FROM landsat_{sensor} WHERE unique_loc_id IN ({placeholders}) AND date>=? AND date<?'''
            params = [*mapping, f"{cfg['year']}-01-01", f"{cfg['year']+1}-01-01"]
            result = con.execute(q, params).fetchall()
            audit[f'sensor_{sensor}_raw'] = audit.get(f'sensor_{sensor}_raw',0)+len(result)
            rows.extend([(*r[:7],mapping[r[7]],r[8]) for r in result])
        if rows:
            monthly, totals = reduce_observations(rows)
            for k,v in totals.items(): audit[k] = audit.get(k,0)+v
            pos = {int(i):begin+j for j,i in enumerate(ids)}
            for ident,month,ndvi,ndmi,evi,count in monthly.iter_rows():
                features[pos[ident],month-1] = [ndvi,ndmi,evi]
                counts[pos[ident],month-1] = count
        if begin % (size*10) == 0:
            log(f'Satellite: {min(begin+size,len(support_ids)):,}/{len(support_ids):,} supports')
    con.close()
    for i in range(len(features)):
        for j in range(3):
            good = np.flatnonzero(np.isfinite(features[i,:,j]))
            # Entirely unobserved supports remain missing; do not manufacture a
            # zero-valued annual satellite trajectory.
            if len(good): features[i,:,j] = np.interp(np.arange(12),good,features[i,good,j])
    audit['supports_without_clear_year'] = int(np.sum(~np.isfinite(features).all(axis=(1,2))))
    audit['missing_support_months_before_interpolation'] = int(np.sum(counts==0))
    dump(out/'satellite_audit.json', audit)
    np.savez_compressed(cache, support_id=support_ids, features=features, clear_count=counts)
    return features, counts


def design(features, parcel, support, area, train, n):
    """Vectorized original bottomup_design; one row per parcel/cell fragment."""
    is_train = np.zeros(n, dtype=bool); is_train[train] = True
    train_values = features[support[is_train[parcel]]]
    lo, hi = np.percentile(train_values, [5,95], axis=0)
    span = hi-lo; span[span==0] = 1
    d = np.empty((12,n,4))
    z = np.clip((features-lo)/span,0,1)
    for m in range(12):
        d[m,:,0] = np.bincount(parcel, weights=area/100, minlength=n)
        for j in range(3):
            d[m,:,j+1] = np.bincount(parcel, weights=area/100*z[support,m,j], minlength=n)
    return d, lo, span


def solve(d,y,alpha):
    # Same augmented-system NNLS as core.ridge_nnls, including intercept penalty.
    return nnls(np.vstack([d,np.sqrt(alpha)*np.eye(d.shape[1])]),
                np.r_[y,np.zeros(d.shape[1])],maxiter=1000)[0]


def fit(d,y,train,alpha):
    return np.stack([solve(d[m,train],y[train,m],alpha) for m in range(12)])


def predict(d,coef):
    return np.einsum('mni,mi->nm',d,coef)


def choose_alpha(cfg, features, parcel, support, area, y, train):
    errors = np.zeros(len(cfg['alphas']))
    count = 0
    splitter = KFold(cfg['inner_folds'],shuffle=True,random_state=cfg['seed'])
    for tri,vai in splitter.split(train):
        tr,va = train[tri],train[vai]
        d,_,_ = design(features,parcel,support,area,tr,len(y))
        for j,a in enumerate(cfg['alphas']):
            p = predict(d,fit(d,y,tr,a))
            errors[j] += np.sum((p[va]-y[va])**2)
        count += len(va)*12
    return cfg['alphas'][int(np.argmin(errors))], (errors/count).tolist()


def scores(y,p):
    delta = p-y
    den = np.linalg.norm(y,axis=1)*np.linalg.norm(p,axis=1)
    cos = np.divide(np.sum(y*p,axis=1), den, out=np.full(len(y),np.nan), where=den>0)
    return dict(monthly_mae_kgal=float(np.mean(abs(delta))),
                monthly_rmse_kgal=float(np.sqrt(np.mean(delta**2))),
                mean_cosine_similarity=float(np.nanmean(cos)),
                cosine_valid_parcels=int(np.isfinite(cos).sum()),
                annual_mae_kgal=float(np.mean(abs(delta.sum(axis=1)))),
                annual_bias_kgal=float(np.mean(delta.sum(axis=1))),
                monthly_mae_by_month_kgal=np.mean(abs(delta),axis=0).tolist())


def add_output_fields(obs, pred):
    result = obs.copy()
    y = result[[f'observed_{m:02d}' for m in range(1,13)]].to_numpy(dtype=float)
    for m in range(12): result[f'predicted_{m+1:02d}'] = pred[:,m]
    for m in range(12): result[f'delta_{m+1:02d}'] = pred[:,m]-y[:,m]
    # np.sum deliberately propagates missing months. A partial sum is not an
    # annual error and must never appear as one.
    result['sum_abs_delta'] = np.sum(abs(pred-y),axis=1)
    result['observed_total'] = np.sum(y,axis=1)
    result['predicted_total'] = np.sum(pred,axis=1)
    result['annual_delta'] = np.sum(pred-y,axis=1)
    return result


def write_gpkg(out, frame, wkb, crs, cfg):
    path = out/'Westminster_2021_observed_predicted.gpkg'
    temporary = out/'Westminster_2021_observed_predicted.tmp.gpkg'
    if temporary.exists(): temporary.unlink()
    pyogrio.raw.write(temporary, wkb, [frame[c].to_numpy() for c in frame],
                     fields=list(frame.columns), driver='GPKG', layer='parcel_water_use',
                     geometry_type='MultiPolygon', crs=crs, promote_to_multi=True)
    con = sqlite3.connect(temporary)
    con.execute('CREATE TABLE analysis_metadata (key TEXT PRIMARY KEY, value TEXT)')
    metadata = dict(year=str(cfg['year']),units='thousand gallons (kgal)',
        model=cfg['model'],prediction_type='held-out parcel five-fold cross-validation',
        delta_definition='predicted minus observed',
        sum_abs_delta_definition='sum(abs(predicted_month - observed_month)) over all 12 months; NULL if incomplete',
        config=json.dumps(cfg),source_manifest=(out/'manifest.json').read_text())
    con.executemany('INSERT INTO analysis_metadata VALUES (?,?)',metadata.items())
    con.execute("INSERT INTO gpkg_contents (table_name,data_type,identifier,description) VALUES ('analysis_metadata','attributes','analysis_metadata','Reproduction and field definitions')")
    con.commit()
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    con.close()
    os.replace(temporary,path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=ROOT/'analysis/config.json')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['year'] != 2021 or cfg['area_unit_m2'] != 100 or cfg['model'] != 'landsat_only_bottomup_nonnegative':
        raise ValueError('This audited implementation supports the configured 2021 NNLS method only')
    out = ROOT/cfg['output_directory']; out.mkdir(parents=True,exist_ok=True)
    paths = [ROOT/cfg['satellite_database'],ROOT/cfg['parcel_database'],Path(__file__),args.config,
             ROOT/'analysis/requirements.lock.txt',ROOT/'analysis/original_src/pipeline.py',
             ROOT/'analysis/original_src/landsat_only.py']
    log('Hashing fixed inputs and implementation')
    inputs = [{'path':str(p.resolve()),'size':p.stat().st_size,'sha256':sha256(p)} for p in paths]
    manifest = {'inputs':inputs,'config':cfg,'python':sys.version}
    previous = out/'manifest.json'
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError('Input, code, environment lock or configuration changed; use a new output directory')
    dump(previous,manifest)
    obs,wkb,crs,parcel,support,area,support_ids = prepare_spatial(cfg,out)
    features,counts = satellite(cfg,out,support_ids)
    y_all = obs[[f'observed_{m:02d}' for m in range(1,13)]].to_numpy(dtype=float)
    missing_features = ~np.isfinite(features).all(axis=(1,2))
    bad_support = np.bincount(parcel,weights=missing_features[support].astype(int),minlength=len(obs)) > 0
    status = np.full(len(obs),'eligible',dtype=object)
    status[bad_support] = 'no_clear_satellite_year'
    status[obs.coverage_fraction.to_numpy()<cfg['minimum_spatial_coverage']] = 'incomplete_satellite_coverage'
    status[~obs.geometry_valid.to_numpy().astype(bool)] = 'invalid_geometry'
    status[np.any(y_all<0,axis=1)] = 'negative_monthly_observation'
    status[~np.isfinite(y_all).all(axis=1)] = 'missing_monthly_observation'
    obs['status'] = status
    eligible = np.flatnonzero(status=='eligible')
    if len(eligible)<cfg['outer_folds']*cfg['inner_folds']:
        raise ValueError('Too few complete eligible parcels')
    obs['observed_months'] = np.isfinite(y_all).sum(axis=1)
    weighted_missing = np.bincount(parcel,weights=area*np.sum(counts[support]==0,axis=1),minlength=len(obs))
    obs['mean_imputed_months'] = np.divide(weighted_missing,np.bincount(parcel,weights=area,minlength=len(obs)),
                                          out=np.full(len(obs),np.nan),where=obs.coverage_fraction.to_numpy()>0)
    obs['support_count'] = np.bincount(parcel,minlength=len(obs))
    mapping = np.full(len(obs),-1,dtype=int); mapping[eligible] = np.arange(len(eligible))
    keep = mapping[parcel]>=0
    ep,es,ea = mapping[parcel[keep]],support[keep],area[keep]
    y = y_all[eligible]
    pred = np.full_like(y,np.nan); baseline = np.full_like(y,np.nan)
    folds = np.full(len(obs),-1,dtype=int)
    fold_models = {}; selections = []
    log(f'Modeling {len(y):,} eligible parcels; statuses: {obs.status.value_counts().to_dict()}')
    outer = KFold(cfg['outer_folds'],shuffle=True,random_state=cfg['seed'])
    splits = list(outer.split(y))
    for fold,(tr,va) in enumerate(splits,1): folds[eligible[va]] = fold
    pd.DataFrame({'parcel_id':obs.parcel_id,'status':status,'fold':folds}).to_csv(out/'fold_assignments.csv',index=False)
    for fold,(tr,va) in enumerate(splits,1):
        log(f'Outer fold {fold}/{cfg["outer_folds"]}: training {len(tr):,}, held out {len(va):,}')
        alpha,errors = choose_alpha(cfg,features,ep,es,ea,y,tr)
        d,lo,span = design(features,ep,es,ea,tr,len(y))
        coef = fit(d,y,tr,alpha)
        pred[va] = predict(d,coef)[va]
        baseline[va] = y[tr].mean(axis=0)
        fold_models.update({f'fold_{fold}_coef':coef,f'fold_{fold}_lo':lo,f'fold_{fold}_span':span})
        selections.append({'fold':fold,'alpha':alpha,'inner_mse_by_alpha':errors})
        log(f'Outer fold {fold} completed; selected alpha {alpha}')
    if not np.isfinite(pred).all(): raise ValueError('Incomplete held-out predictions')
    alpha,errors = choose_alpha(cfg,features,ep,es,ea,y,np.arange(len(y)))
    d,lo,span = design(features,ep,es,ea,np.arange(len(y)),len(y))
    coef = fit(d,y,np.arange(len(y)),alpha)
    np.savez_compressed(out/'model_state.npz',full_coef=coef,full_lo=lo,full_span=span,
                         full_alpha=alpha,**fold_models)
    dump(out/'model_selection.json',{'outer':selections,'full_alpha':alpha,'full_inner_mse_by_alpha':errors})
    all_pred = np.full_like(y_all,np.nan); all_pred[eligible] = pred
    obs['fold'] = folds
    frame = add_output_fields(obs,all_pred)
    # Keep the user's requested monthly groups adjacent and ordered.
    fields = ['parcel_id','parcel_objectid','parcel_apn']
    fields += [f'{prefix}_{m:02d}' for prefix in ['observed','predicted','delta'] for m in range(1,13)]
    fields += ['sum_abs_delta']
    fields += [c for c in frame if c not in fields]
    frame = frame[fields]
    frame.to_csv(out/'parcel_predictions.csv',index=False)
    pd.DataFrame(baseline,index=obs.parcel_id.to_numpy()[eligible],
                 columns=[f'predicted_{m:02d}' for m in range(1,13)]).rename_axis('parcel_id').to_csv(out/'mean_curve_reference.csv')
    difference = np.mean(abs(baseline-y),axis=1)-np.mean(abs(pred-y),axis=1)
    rng = np.random.default_rng(cfg['seed'])
    bootstrap = np.array([difference[rng.integers(0,len(y),len(y))].mean()
                          for _ in range(cfg['parcel_bootstrap_draws'])])
    metrics = {'eligible_parcels':len(y),'retained_parcels':len(obs),
               'status_counts':obs.status.value_counts().to_dict(),
               'model':scores(y,pred),'mean_curve_reference':scores(y,baseline),
               'mean_mae_improvement_over_reference_kgal':float(difference.mean()),
               'paired_parcel_bootstrap_95_interval_kgal':np.quantile(bootstrap,[.025,.975]).tolist()}
    dump(out/'metrics.json',metrics)
    path = write_gpkg(out,frame,wkb,crs,cfg)
    log(f'Finished: {path}; monthly MAE {metrics["model"]["monthly_mae_kgal"]:.3f} kgal')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
