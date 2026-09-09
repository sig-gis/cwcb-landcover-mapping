# Recorded review inputs

These files record the 53 circuits whose maps were accepted in v7. They are the explicit site-selection decisions for the revised process. They contain no model predictions, generated parcel geometry, final route geometry, rankings or maps.

- `parcels.csv`: circuit ID, source parcel ID, assessment/context flag and reviewed side of the street. These are the membership decisions. Source parcel IDs are feature IDs in `parcels_cow_summary`.
- `circuits.json`: circuit name, intended category, source street IDs, partial street legs, park parcel ID, suggested start and review reason. A road fraction runs from 0 at the source line's first coordinate to 1 at its last. The park access is the point on a source street from which the short boundary connection is drawn. These references let the build retrieve original geometry rather than reuse a finished map.
- `sources.json`: paths and hashes for the original parcel/account inventory, municipal streets and fixed model predictions. The build stops if these change.

The recorded crossing-exposure values preserve the review's simple priority: 1 for a cul-de-sac without a through-junction, 2 for another street unit without a through-junction, 3 for a cul-de-sac with one through-junction and 4 for another unit with one through-junction. They are ordinal planning priorities, not measured numbers of pedestrian crossings.

`record_review.py` documents the one-time transcription from the accepted v7 result into these source-ID records. It was run once. It is not part of the revised build and must not be rerun to manufacture a new selection automatically. The historic review used the v7 machinery; the revised workflow makes its accepted decisions explicit and removes that machinery from subsequent builds. The review notes describe the historic layout decisions; any references there to former ranks or primary/backup status do not determine the revised ranking.

Run `scripts/field_sampling.py` to rebuild from these records and the three source datasets. It imports only its drawing module, `scripts/field_sampling_maps.py`, in addition to installed libraries. Neither module reads v7 outputs or invokes the old candidate search, frontage heuristics, ranking adjustment or finalization scripts.

The remaining geometry operations serve drawing and walking: close source endpoint gaps within 1.5 m, retrieve the specified street pieces, connect the recorded access point to the paired parcel boundary, traverse the connected street network outward and back, and number parcels around its perimeter. An 8 m corridor is used only to order parcels around bends and cul-de-sac bulbs. It is not a mapped sidewalk or an additional site-selection criterion.

The revision retains the accepted parcel membership, paired spaces and street-walk geometry. Ranking is recalculated with the shorter rule in METHODS.md, so ranks and primary choices can differ from v7. Stable `circuit_id` values identify sites independently of category rank.

To test the delivered output again, from the project folder run:

```bash
MPLCONFIGDIR=/tmp/cwcb-matplotlib .venv/bin/python scripts/field_sampling.py --check-existing
```

This generates every deliverable in a new temporary directory and compares it with `outputs/street_samples_revised`. The normal build command in METHODS.md can instead write a new permanent output directory. The only read of an existing result is the verification comparison, after the fresh build has completed.

Verification on 2026-09-09: all 53 circuit memberships and category assignments match v7; all 661 mapped parcel geometries match exactly; all street-walk geometries have zero Hausdorff distance from their accepted counterparts. The simpler ranking changed 36 ranks. Numbering in C34988 and C33650 rotates the same perimeter sequence by one parcel. Two complete fresh builds matched all GIS fields and geometry exactly; the CSV, detailed PDF, both PNGs, methods and run record were byte-for-byte identical. GeoPackage creation timestamps were not compared.
