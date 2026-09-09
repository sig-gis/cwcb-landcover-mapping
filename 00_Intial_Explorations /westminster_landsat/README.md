# Westminster Landsat water-use analysis and field sampling

This is the expanded 2021 Westminster analysis: load Landsat 8/9 exports in
batches, estimate twelve monthly metered totals with the existing nonnegative
Landsat model fitted separately by parcel size, and rebuild the reviewed
neighborhood field sample. Indoor/Outdoor estimates are not used.

Methods: [seasonal model](analysis/METHODS.md),
[size-class fits](analysis/size_binned/METHODS.md), and
[field sampling](analysis/field_sampling/METHODS.md).
The sampling inputs record individual review decisions for 53 circuits and
657 assessment parcels. The script rebuilds those choices; it does not select
the neighborhoods independently. It generates geometry, walks, categories,
rankings, a GeoPackage, a CSV, detailed PDF maps and both overview PNGs.

## Setup and inputs

Run commands from this directory. The recorded environment used Python 3.14.4.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r analysis/size_binned/requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -v
```

Source data and generated GIS/water-use tables are excluded from Git. Supply
these files at the following paths, relative to this directory:

| Input | Path |
|---|---|
| Parcel geometry and joined 2021 monthly accounts | `prior analysis/add_par_all/Parcels_CoW_2021.gpkg` |
| Loaded Landsat 8/9 database | `outputs/CWCB_Westminster_2026_08_19_1700.sqlite` |
| Reviewed municipal street-centerline snapshot | `outputs/field_review_evidence/routable_streets.geojson` |

The parcel input must contain `parcels_cow_summary` and `cow_accounts_joined`;
the seasonal methods describe the fields and missing-data rules. Street IDs
refer to Westminster's `Transportation/StreetCenterlines/MapServer/0` export.
Use the original snapshot: fetching a newer street layer can change IDs or
geometry. [sources.json](analysis/field_sampling/sources.json) records the exact
parcel, street and model-prediction hashes required by the reviewed sample.

## Run

If the satellite database is not already available, load the original CSV export
folder (only files named `Landsat8_*.csv` and `Landsat9_*.csv` are read):

```bash
.venv/bin/python scripts/load_landsat.py --source /path/to/landsat_csvs --batch-size 100000
```

The loader commits bounded batches, records completed files for resume, checks
identities and coordinates, and removes duplicate observations. The source
folder must remain unchanged when resuming a database.

Run the shared fit, validate its saved outputs, then fit the fixed size classes:

```bash
.venv/bin/python scripts/analyze_seasonal.py
.venv/bin/python scripts/validate_seasonal.py
.venv/bin/python scripts/analyze_size_binned.py
```

Validation records the artifact hashes consumed by the size-class fit. Both
models retain held-out monthly predictions; full-data coefficients are saved
separately. Existing runs have input manifests, so use fresh output directories
when moving or changing a run. The scripts accept `--config` for alternative
paths; the validator checks the default 2021 output paths.

To reproduce the reviewed sample, provide the exact size-binned
`outputs/seasonal_2021_size_binned/parcel_predictions.csv` recorded in
`sources.json`, together with its parcel and street inputs. Then run:

```bash
MPLCONFIGDIR=/tmp/cwcb-matplotlib .venv/bin/python scripts/field_sampling.py --verify
```

The output directory must not exist. Verification builds everything a second
time in an empty temporary directory and compares GIS attributes and geometry,
CSV, PDF, both PNGs and run records. It excludes GeoPackage creation timestamps.
Exact historical sampling reproduction requires the recorded input bytes;
a different model result needs a deliberate new review/version, not a bypass
of the hash check.

The original review transcription script is retained for provenance only. It
requires the local v7 result, which is not included here, and is not needed to
run the revised process. The earlier pilot source in `analysis/original_src`
is retained for the model-equivalence tests, not as another pipeline to run.

## Overview maps

These are the accepted revised-run figures; the build regenerates both under
`outputs/street_samples_revised/`.

[Top-ranked streets](figures/top_ranked_streets.png) ·
[All sites overview](figures/all_sites_overview.png)

Checked on 2026-09-09 from this repository copy: eight loader/model tests passed;
the saved seasonal output passed observation, geometry and held-out prediction
reconstruction checks; two complete sampling builds matched. Model fitting and
the full satellite import were not rerun for this commit.
