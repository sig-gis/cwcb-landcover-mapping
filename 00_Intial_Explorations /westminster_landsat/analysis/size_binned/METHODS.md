# Version 2: fixed parcel-size classes

The user authorized separate fits in six fixed size classes: <500, 500–1,000,
1,000–2,000, 2,000–10,000, 10,000–50,000, and >=50,000 m². Lower bounds are
inclusive and upper bounds exclusive. Breaks were chosen from the area histogram
before this fit, not optimized against its errors.

Reuse the original verified satellite-month cache, parcel/pixel intersections,
raw monthly metered targets, eligibility decisions and outer five-fold parcel
assignments. Check their hashes against the first run's artifact checksums.
No Indoor/Outdoor estimate is used. Sources and version 1 outputs remain intact.

Within each class, use exactly the original nonnegative Landsat-only equation,
area units, NDVI/NDMI/EVI predictors, percentile scaling and alpha grid. Each
outer model uses only the other folds within its size class. Inner five-fold
selection and all learned scaling are restricted to class-specific training
parcels. Use the same seed, 20260903, and grid [0.1,1,10,100,1000]. Save one
full-data fit per size class in addition to every class/fold model. The mapped
predictions are exclusively held-out predictions. Save class/fold training counts.

The same training-class mean monthly curve is evaluated as a reference; the
original global model is compared on exactly the same eligible parcels/folds.
Report error, annual-volume bias, fraction underpredicted and relative departure
for every class and overall. Do not tune size boundaries after inspecting results.

This is a revision informed by the first evaluation, not a fresh untouched test
set. It evaluates unseen parcels in the same city/year; neighboring parcels can
share satellite supports. The largest class has only 63 parcels and remains
heterogeneous. Separate fits can create discontinuities at class boundaries.

GeoPackage: outputs/seasonal_2021_size_binned/Westminster_2021_size_binned.gpkg.
Layer: parcel_water_use. Preserve the 12 observed/predicted/delta fields and
sum_abs_delta. Add size_class_id (0–5), size_class, signed_departure (observed
minus predicted annual total, kgal) and relative_departure ((observed-predicted)
/(observed+predicted), NULL if denominator is zero or inputs are missing).
Monthly delta and annual_delta retain version 1's predicted-minus-observed sign.
Excluded parcels remain with NULL predictions and their existing status.

Reproduce from the project directory:

```
.venv/bin/python -m pip install -r analysis/size_binned/requirements.lock.txt
.venv/bin/python scripts/analyze_size_binned.py --config analysis/size_binned/config.json
```

The output manifest records exact reused artifacts, code, config, methods and
environment lock. The run verifies all saved monthly values, departure fields,
geometry, IDs, fold preservation, SQLite integrity and foreign keys. It also
independently reconstructs every held-out prediction by summing fragment rates
from the saved class/fold coefficients and scalers, rather than refitting.
