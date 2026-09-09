"""Build the reviewed field sample from source IDs; no candidate-selection engine."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyogrio.raw
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.ops import substring

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'analysis/field_sampling'
CRS = 'EPSG:32613'
CATEGORIES = ['above_prediction', 'below_prediction', 'mixed', 'close_to_prediction']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_layer(path, layer, columns=None):
    meta, ids, wkb, values = pyogrio.raw.read(
        path, layer=layer, columns=columns, return_fids=True)
    frame = pd.DataFrame(dict(zip(meta['fields'], values)), index=ids)
    return frame, shapely.from_wkb(wkb), meta['crs']


def source_roads(path):
    """Read centerlines and close source endpoint gaps within 1.5 metres."""
    features = json.loads(Path(path).read_text())['features']
    transform = Transformer.from_crs(4326, CRS, always_xy=True)
    roads = {f['properties']['OBJECTID']: shapely.transform(
        shapely.geometry.shape(f['geometry']), transform.transform, interleaved=False)
        for f in features}
    public = [f['properties']['OBJECTID'] for f in features
              if f['properties']['STREETCLASSIFICATION'] in
              ['Local', 'Collector', 'Major Arterial', 'Minor Arterial',
               'Highway', 'Interstate', 'Ramp'] and roads[f['properties']['OBJECTID']].length > 1]
    xy = np.array([roads[k].coords[j] for k in public for j in [0, -1]])
    parent = np.arange(len(xy))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in sorted(cKDTree(xy).query_pairs(1.5)):
        parent[root(a)] = root(b)
    groups = {}
    for i in range(len(xy)):
        groups.setdefault(root(i), []).append(i)
    for indices in groups.values():
        xy[indices] = xy[indices].mean(axis=0)
    for i, key in enumerate(public):
        coords = np.array(roads[key].coords)
        coords[0], coords[-1] = xy[2*i], xy[2*i+1]
        roads[key] = shapely.LineString(coords)
    return roads


def segment(reference, roads):
    return substring(roads[reference['road_id']], reference.get('from', 0),
                     reference.get('to', 1), normalized=True)


def closed_walk(network, start):
    """Visit each street edge out and back, returning to the recorded start."""
    pieces = []
    for part in shapely.get_parts(network):
        at = part.project(start)
        if part.distance(start) < .001 and 0 < at < part.length:
            pieces.extend([substring(part, 0, at), substring(part, at, part.length)])
        else:
            pieces.append(part)
    graph = {}
    for part in pieces:
        coords = np.round(shapely.get_coordinates(part), 4)
        for a, b in zip(coords[:-1], coords[1:]):
            a, b = tuple(a), tuple(b)
            if a != b:
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
    origin = min(graph, key=lambda p: shapely.Point(p).distance(start))
    visited, trace = set(), [origin]
    stack = [(origin, iter(sorted(graph[origin])))]
    while stack:
        node, neighbors = stack[-1]
        try:
            other = next(neighbors)
        except StopIteration:
            stack.pop()
            if stack:
                trace.append(stack[-1][0])
            continue
        edge = tuple(sorted((node, other)))
        if edge not in visited:
            visited.add(edge)
            trace.append(other)
            stack.append((other, iter(sorted(graph[other]))))
    if len(visited) != sum(map(len, graph.values())) // 2:
        raise ValueError('The recorded street legs are disconnected')
    route = shapely.LineString(trace)
    assert route.is_closed
    assert network.difference(route.buffer(.002)).is_empty
    assert route.difference(network.buffer(.002)).is_empty
    return route


def statistics(group):
    assessed = group[group.assessment_required]
    modeled = assessed[assessed.model_status.eq('eligible') &
                       np.isfinite(assessed.relative_departure)]
    if len(modeled) < 4 or len(modeled) / len(assessed) < .8:
        raise ValueError('Insufficient model coverage for a reviewed circuit')
    r = modeled.relative_departure
    positive, negative = (r > 0).mean(), (r < 0).mean()
    above, below = (r >= .2).mean(), (r <= -.2).mean()
    close = ((r.abs() <= .1) & (modeled.seasonal_mismatch <= .2)).mean()
    if positive >= .8 and above >= .5:
        category, fit = CATEGORIES[0], above
    elif negative >= .8 and below >= .5:
        category, fit = CATEGORIES[1], below
    elif above >= .25 and below >= .25:
        category, fit = CATEGORIES[2], 2 * min(above, below)
    elif close >= .6:
        category, fit = CATEGORIES[3], close
    else:
        raise ValueError('Reviewed circuit no longer meets a category definition')
    return dict(category=category, category_fit=float(fit),
                assessment_parcels=len(assessed), modeled_parcels=len(modeled),
                model_coverage=len(modeled)/len(assessed),
                above_fraction=float(above), below_fraction=float(below),
                close_fraction=float(close), median_departure=float(r.median()),
                observed_total=float(modeled.observed_total.sum()),
                predicted_total=float(modeled.predicted_total.sum()))


def write_layer(path, name, frame, geometry, kind):
    values = [frame[c].to_numpy() for c in frame]
    values = [v.astype('int32') if v.dtype.kind == 'b' else v for v in values]
    pyogrio.raw.write(path, shapely.to_wkb(geometry), values, fields=list(frame),
                      layer=name, driver='GPKG', crs=CRS, geometry_type=kind,
                      promote_to_multi=kind.startswith('Multi'))


def build(out):
    if out.exists():
        raise FileExistsError(f'Use a new output directory; existing results are preserved: {out}')
    sources = json.loads((DESIGN / 'sources.json').read_text())
    for record in sources.values():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError(f"Source has changed: {record['path']}")
    choices = json.loads((DESIGN / 'circuits.json').read_text())
    members = pd.read_csv(DESIGN / 'parcels.csv')
    assert members.parcel_id.is_unique
    assert set(members.circuit_id) == {c['circuit_id'] for c in choices}
    source = ROOT / sources['parcels']['path']
    inventory, geometry, crs = read_layer(source, 'parcels_cow_summary',
                                        ['FACILITYNAME', 'CITYOWN', 'HOA'])
    transform = Transformer.from_crs(crs, CRS, always_xy=True)
    geometry = shapely.transform(geometry, transform.transform, interleaved=False)
    parcel_geometry = dict(zip(inventory.index, geometry))
    roads = source_roads(ROOT / sources['roads']['path'])
    with sqlite3.connect(f'{source.as_uri()}?mode=ro', uri=True) as con:
        addresses = pd.read_sql_query(
            'SELECT parcel_fid,cow_service_address FROM cow_accounts_joined', con)
    addresses = addresses.groupby('parcel_fid').cow_service_address.agg(
        lambda values: ' | '.join(sorted(set(str(v) for v in values if pd.notna(v)))))
    model = pd.read_csv(ROOT / sources['model']['path'])
    model = model[['parcel_id', 'status', 'observed_total', 'predicted_total',
                   'sum_abs_delta']].rename(columns={'status': 'model_status'})
    members = members.merge(model, on='parcel_id', how='left', validate='one_to_one')
    total = members.observed_total + members.predicted_total
    members['relative_departure'] = (members.observed_total - members.predicted_total) / total
    members['seasonal_mismatch'] = members.sum_abs_delta / total
    members['service_addresses'] = members.parcel_id.map(addresses).fillna('')
    members['assessment_required'] = members.assessment_required.astype(bool)
    members['visit_order'] = 0
    sites, cores, walks, routes, starts, parks = [], [], [], [], [], []
    reverse = Transformer.from_crs(CRS, 4326, always_xy=True)
    for choice in choices:
        cid = choice['circuit_id']
        core = shapely.union_all([roads[k] for k in choice['core_road_ids']])
        legs = [segment(r, roads) for r in choice['park_road_legs']]
        park = parcel_geometry[choice['park_parcel_id']]
        access = choice['park_access']
        road_point = roads[access['road_id']].interpolate(access['fraction'], normalized=True)
        legs.append(shapely.shortest_line(road_point, park))
        park_walk = shapely.union_all([g for g in legs if g.length > 0])
        network = shapely.union_all([core, park_walk])
        staging = choice['start']
        start = roads[staging['road_id']].interpolate(staging['fraction'], normalized=True)
        route = closed_walk(network, start)
        ix = members.index[members.circuit_id.eq(cid)]
        group = members.loc[ix]
        # The two sides are recorded as part of the reviewed parcel membership.
        qualified = group[group.assessment_required & group.model_status.eq('eligible') &
                          np.isfinite(group.relative_departure)]
        assert set(qualified.reviewed_side) == {'left', 'right'}, cid
        stats = statistics(group)
        if stats['category'] != choice['intended_category']:
            raise ValueError(f'{cid}: category changed; review the source values')
        # Order the houses consecutively around the corridor, including bulb fans.
        corridor = network.buffer(8, quad_segs=32)
        assert corridor.geom_type == 'Polygon', cid
        ring = shapely.LineString(corridor.exterior.coords)
        if shapely.is_ccw(corridor.exterior):
            ring = shapely.reverse(ring)
        origin = ring.project(start)
        order = sorted(group.index[group.assessment_required], key=lambda j: (
            (ring.project(parcel_geometry[members.at[j, 'parcel_id']].representative_point())
             - origin) % ring.length, members.at[j, 'parcel_id']))
        members.loc[order, 'visit_order'] = np.arange(1, len(order)+1)
        lon, lat = reverse.transform(start.x, start.y)
        facility = inventory.loc[choice['park_parcel_id'], 'FACILITYNAME']
        named = isinstance(facility, str) and bool(facility.strip())
        sites.append(dict(circuit_id=cid, street_name=choice['street_name'], **stats,
                          crossing_exposure=choice['crossing_exposure'],
                          estimated_walk_m=route.length, park_id=choice['park_parcel_id'],
                          park_name=facility if named else f"Common tract {choice['park_parcel_id']}",
                          park_area_m2=park.area, latitude=lat, longitude=lon,
                          review_note=choice['review_note']))
        cores.append(core); walks.append(park_walk); routes.append(route)
        starts.append(start); parks.append(park)
    sites = pd.DataFrame(sites)
    rank = sites.sort_values(['crossing_exposure', 'category_fit', 'modeled_parcels',
                             'circuit_id'], ascending=[True, False, False, True])
    ranking = pd.Series(rank.groupby('category').cumcount().to_numpy()+1, index=rank.circuit_id)
    sites['rank_in_category'] = sites.circuit_id.map(ranking)
    sites['priority'] = np.where(sites.rank_in_category <= 3, 'primary', 'backup')
    sites['site_id'] = sites.category + '_' + sites.rank_in_category.map(lambda n: f'{n:02d}')
    for column in ['site_id', 'category', 'rank_in_category', 'priority']:
        members[column] = members.circuit_id.map(sites.set_index('circuit_id')[column])
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'ranked_street_samples_with_spaces.gpkg'
    write_layer(path, 'crew_circuits', sites, routes, 'LineString')
    write_layer(path, 'ranked_streets', sites, cores, 'MultiLineString')
    write_layer(path, 'park_walk_connections', sites[['circuit_id', 'site_id']], walks, 'MultiLineString')
    write_layer(path, 'staging_references', sites[['circuit_id', 'site_id']], starts, 'Point')
    write_layer(path, 'paired_spaces', sites[['circuit_id', 'site_id', 'park_id', 'park_name',
                                            'park_area_m2']], parks, 'MultiPolygon')
    write_layer(path, 'sample_parcels', members,
                [parcel_geometry[k] for k in members.parcel_id], 'MultiPolygon')
    sites.sort_values(['category', 'rank_in_category']).to_csv(out / 'circuit_summary.csv', index=False)
    from field_sampling_maps import render
    render(out, sites, members, cores, walks, starts, parks, geometry, parcel_geometry, roads)
    with sqlite3.connect(path) as con:
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
    report = dict(circuits=len(sites), mapped_parcels=len(members),
                  assessment_parcels=int(members.assessment_required.sum()),
                  category_counts=sites.category.value_counts().to_dict(),
                  all_routes_closed=True, all_route_legs_covered=True,
                  inputs={str(p.relative_to(ROOT)): digest(p) for p in [
                      DESIGN/'sources.json', DESIGN/'circuits.json', DESIGN/'parcels.csv',
                      DESIGN/'METHODS.md',
                      Path(__file__), Path(__file__).with_name('field_sampling_maps.py')]},
                  source_files=sources,
                  library_versions={name: importlib.metadata.version(name) for name in
                                    ['numpy', 'pandas', 'shapely', 'pyogrio', 'pyproj',
                                     'scipy', 'matplotlib']})
    shutil.copyfile(DESIGN/'METHODS.md', out/'METHODS.md')
    (out / 'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Built {len(sites)} circuits and all maps in {out}', flush=True)


def verify(out):
    """Rebuild in an empty directory and compare every generated deliverable."""
    with tempfile.TemporaryDirectory(prefix='cwcb-sampling-') as temporary:
        repeated = Path(temporary)/'repeated'
        build(repeated)
        products = ['circuit_summary.csv', 'top_ranked_streets.png',
                    'all_sites_overview.png', 'ranked_field_review_maps.pdf', 'run.json', 'METHODS.md']
        hashes = {}
        for name in products:
            hashes[name] = digest(out/name)
            assert hashes[name] == digest(repeated/name), f'Rebuild differs: {name}'
        a = out/'ranked_street_samples_with_spaces.gpkg'
        b = repeated/a.name
        layers = ['crew_circuits', 'ranked_streets', 'park_walk_connections',
                  'staging_references', 'paired_spaces', 'sample_parcels']
        for layer in layers:
            frame_a, geometry_a, crs_a = read_layer(a, layer)
            frame_b, geometry_b, crs_b = read_layer(b, layer)
            pd.testing.assert_frame_equal(frame_a, frame_b)
            assert crs_a == crs_b
            assert np.array_equal(shapely.to_wkb(geometry_a), shapely.to_wkb(geometry_b)), layer
        report = dict(fresh_full_build_matches=True, identical_file_hashes=hashes,
                      geopackage_layers_with_identical_fields_and_geometry=layers,
                      note='GeoPackage creation timestamps are not compared.')
        (out/'reproduction_check.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Fresh rebuild matches: all tables, all geometry, PDF, CSV and both PNGs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/street_samples_revised')
    parser.add_argument('--verify', action='store_true', help='Run a second fresh build and compare all outputs')
    parser.add_argument('--check-existing', action='store_true', help='Compare an existing build with a fresh rebuild')
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check_existing:
        build(output)
    if args.verify or args.check_existing:
        verify(output)
