"""One-time transcription of the accepted v7 GIS review into source-ID decisions.

This is not called by the revised build. It records the provenance of the
selection inputs instead of pretending that the new script selected these sites.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from field_sampling import source_roads, read_layer, digest


def main():
    out = Path(__file__).parent
    baseline = ROOT/'outputs/street_samples_v7/ranked_street_samples_with_spaces.gpkg'
    sources = {
        'parcels': 'prior analysis/add_par_all/Parcels_CoW_2021.gpkg',
        'model': 'outputs/seasonal_2021_size_binned/parcel_predictions.csv',
        'roads': 'outputs/field_review_evidence/routable_streets.geojson',
    }
    if (out/'circuits.json').exists():
        raise RuntimeError('Review inputs already recorded; do not overwrite them')
    sources = {k: dict(path=v, sha256=digest(ROOT/v)) for k, v in sources.items()}
    roads = source_roads(ROOT/sources['roads']['path'])
    road_ids = list(roads)
    road_geometry = np.array(list(roads.values()), dtype=object)
    tree = shapely.STRtree(road_geometry)
    sites, cores, _ = read_layer(baseline, 'ranked_streets')
    members, _, _ = read_layer(baseline, 'sample_parcels')
    walks, walk_geometry, _ = read_layer(baseline, 'park_walk_connections')
    pairs, park_geometry, _ = read_layer(baseline, 'paired_spaces')
    decisions = []
    mapping = {}
    for i, site in sites.iterrows():
        cid = site.candidate_id.removeprefix('unit_').removesuffix('_0')
        cid = 'C' + cid
        mapping[site.site_id] = cid
        core_ids = [int(v) for v in site.source_street_ids.split(',')]
        park = park_geometry[np.flatnonzero(pairs.site_id.eq(site.site_id))[0]]
        walk = walk_geometry[np.flatnonzero(walks.site_id.eq(site.site_id))[0]]
        road_legs, offroad = [], []
        for part in shapely.get_parts(walk):
            for a, b in zip(part.coords[:-1], part.coords[1:]):
                edge = shapely.LineString([a, b])
                if edge.length < .00001:
                    continue
                near = tree.query(edge.interpolate(.5, normalized=True),
                                  predicate='dwithin', distance=.001)
                matches = [j for j in near if edge.difference(road_geometry[j].buffer(.001)).length < .001]
                if matches:
                    j = min(matches, key=lambda k: (edge.hausdorff_distance(road_geometry[k]), road_ids[k]))
                    road = road_geometry[j]
                    ref = dict(road_id=road_ids[j],
                               **{'from': road.project(shapely.Point(a), normalized=True),
                                  'to': road.project(shapely.Point(b), normalized=True)})
                    if road_legs and road_legs[-1]['road_id'] == ref['road_id'] and abs(road_legs[-1]['to']-ref['from']) < 1e-8:
                        road_legs[-1]['to'] = ref['to']
                    else:
                        road_legs.append(ref)
                else:
                    offroad.append(edge)
        assert len(offroad) == 1, (site.site_id, len(offroad))
        access_edge = offroad[0]
        endpoints = [shapely.Point(access_edge.coords[0]), shapely.Point(access_edge.coords[-1])]
        access_point = max(endpoints, key=lambda p: p.distance(park))
        road_i = int(tree.nearest(access_point))
        access = dict(road_id=road_ids[road_i],
                      fraction=road_geometry[road_i].project(access_point, normalized=True))
        assert road_geometry[road_i].distance(access_point) < .001
        start_point = shapely.Point(site.staging_x, site.staging_y)
        start_id = min(core_ids, key=lambda k: (roads[k].distance(start_point), k))
        start = dict(road_id=start_id, fraction=roads[start_id].project(start_point, normalized=True))
        decisions.append(dict(circuit_id=cid, street_name=site.street_name,
                              intended_category=site.category, core_road_ids=core_ids,
                              park_road_legs=road_legs, park_parcel_id=int(site.pair_id),
                              park_access=access, start=start,
                              crossing_exposure=int(site.crossing_exposure_score),
                              review_note=site.individual_reason))
    members['circuit_id'] = members.site_id.map(mapping)
    members['assessment_required'] = members.assessment_required.astype(bool)
    members = members.rename(columns={'street_side': 'reviewed_side'})
    members[['circuit_id', 'parcel_id', 'assessment_required', 'reviewed_side']].sort_values(
        ['circuit_id', 'parcel_id']).to_csv(out/'parcels.csv', index=False)
    (out/'circuits.json').write_text(json.dumps(sorted(decisions, key=lambda d: d['circuit_id']), indent=2)+'\n')
    (out/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(f'Recorded {len(decisions)} reviewed circuits as source IDs and road fractions')


if __name__ == '__main__':
    main()
