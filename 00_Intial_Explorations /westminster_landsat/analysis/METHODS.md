# Westminster 2021 seasonal-curve spatial output

## Fixed scope

The target is the twelve raw monthly metered totals, in thousand gallons (kgal).
The user explicitly excluded the source Indoor/Outdoor estimates. Neither is
read or used as a target or predictor. Only Landsat 8/9 is used for satellite
features. The year is 2021 because these are the supplied labeled water records.

The retained method is `landsat_only_bottomup_nonnegative` from the existing
analysis. The original `pipeline.py` and `landsat_only.py` were located at
`/home/lime/Documents/CWCB_Local_Analysis/Other Analsyses/AMZ_wateruse_demo/water/water_curve_reset/src`
and copied unchanged to `analysis/original_src/` for provenance. This run does
not introduce new model families. The mean monthly training curve is evaluated
on the same held-out parcels as a reference, not selected as a replacement.

## Inputs and spatial construction

The satellite source is the completed local SQLite database specified in
`config.json`. Parcel geometry comes from `parcels_cow_summary` in the supplied
citywide GeoPackage. Monthly totals are reconstructed from `cow_accounts_joined`
grouped by parcel FID. If any assigned account lacks a month's reading, that
parcel-month remains NULL. Account IDs and customer details are not exported.
Supplied matches are retained, including address conflicts, with their counts
flagged. This run does not redo the account join.

All 30,912 parcels with assigned accounts are retained in the output. Parcels
with incomplete/nonfinite or negative monthly readings, invalid geometry,
incomplete satellite-grid coverage, or a contributing support with no clear
observations all year are not fitted or scored. Their status explains why;
their predictions and annual errors remain NULL. Valid zero observations are
retained. No consumption outlier trimming is applied.

Areas are in EPSG:32613. The exported Landsat 8 support coordinates are checked
to form a unique regular 30 m UTM grid. Square cells centered at these supports
are intersected with parcels; positive intersection areas supply the weights.
Eleven Landsat 8 IDs differ only by coordinate roundoff at an existing grid
center. Consolidate physical grid centers and remove repeated identical
image/band observations after remapping IDs, preserving distinct image records.
Save the sensor/source-ID/canonical-support mapping in `support_aliases.csv`.
Coverage must be at least 0.999999 of the parcel area. Geometry is preserved in
the source output CRS, with Polygon promoted to MultiPolygon where necessary.

**Spatial adaptation:** the pilot used aerial-object/parcel fragments and each
object's nearest Landsat support. Citywide aerial objects are unavailable here.
This run uses parcel/Landsat-cell intersections, retaining the same additive
area-weighted equation. It is an extension of the model, not an exact replay of
the pilot's spatial construction. Percentile scaling weights each intersection
as one fragment, as the original implementation weighted each aerial fragment.

## Landsat features

For 2021, fetch observations in batches of at most 500 requested support IDs per
sensor, using the source `(unique_loc_id,date)` index. Sensor IDs are independent;
match locations across sensors by their physical UTM grid centers.
Scale SR_B2/B4/B5/B6 by `DN * 0.0000275 - 0.2`. Exclude missing required bands or
QA_PIXEL, any QA_PIXEL bit 0–5, and nonzero QA_RADSAT. As in the original code,
NULL QA_RADSAT is accepted if the remaining required fields exist.

Compute NDVI, NDMI and EVI using the original equations. Reject nonfinite indices
or absolute NDVI/NDMI greater than one. Pool valid observations from both sensors
before computing per-support monthly medians. Linearly interpolate missing
months within each support and carry endpoint values. The new Landsat 9 2021
export has no reflectance bands, so those rows cannot contribute. The older pilot
database is not silently merged into the new citywide inputs.

**Missing-feature adaptation:** unlike the pilot code's zero fallback, a support
with no clear observations for the entire year remains missing; affected parcels
are not scored. The output records support counts and area-weighted counts of
interpolated months. Interpolation uses the full contemporaneous 2021 satellite
trajectory; this is retrospective estimation, not a forecast using only past
months.

## Model and validation

For fragment j and month m, `z` is NDVI/NDMI/EVI scaled by training-fragment
5th/95th percentiles and clipped to [0,1]. A zero percentile span becomes one.
The monthly rate in kgal per 100 square meters is

`rate[j,m] = beta[0,m] + beta[1:4,m] dot z[j,m]`, with every beta nonnegative.

Sum `area[j]/100 * rate[j,m]` across a parcel. Fit the same nonnegative least
squares objective with ridge penalty on all four coefficients, including the
intercept. Choose one alpha shared by all months from the original Landsat
grid `[0.1, 1, 10, 100, 1000]` using minimum inner-validation mean squared error.
The vectorized design and solver are tested against functions extracted directly
from the archived original source.

**Validation adaptation:** fixed five-fold outer parcel validation and five-fold
inner parcel validation replace the pilot's nested leave-one-parcel-out loops,
which are impractical for approximately 30,000 parcels. Both use shuffle=True
and seed 20260903. Sort parcels by source FID before assigning folds. All twelve
months of a parcel stay together. Save outer assignments; inner assignments are
deterministically reconstructed from sorted outer-training rows and the seed.
Fit scalers separately on each inner training set as well as each outer training
set. The original bottom-up tuning held the outer-training scaler fixed during
inner tuning; this implementation excludes inner-validation features as well.

This is validation on unseen parcels in the same year and city. Nearby parcels
may share Landsat supports across folds, as in the pilot. It does not establish
performance in an unseen neighborhood or year. Outer-validation readings are
never used to fit or select the model predicting that parcel.

After recording held-out predictions, fit and save a model on all eligible
parcels using the same inner selection. This final model does not replace the
held-out predictions in the GeoPackage.

## Deliverables and meanings

Primary file: `outputs/seasonal_2021/Westminster_2021_observed_predicted.gpkg`.
Spatial layer: `parcel_water_use`; attribute layer: `analysis_metadata`.

| Fields | Meaning |
| --- | --- |
| observed_01 through observed_12 | January–December account-summed metered totals |
| predicted_01 through predicted_12 | Held-out monthly model predictions |
| delta_01 through delta_12 | Predicted minus observed: positive means overprediction |
| sum_abs_delta | Sum of twelve absolute monthly errors, kgal |
| observed_total, predicted_total | Sums over all twelve months |
| annual_delta | Predicted annual sum minus observed annual sum |
| fold | Outer validation fold 1–5; -1 means not evaluated |
| status | Eligibility or exclusion reason |
| conflict_accounts | Number of supplied matches with an address conflict |
| observed_months | Number of complete account-summed monthly observations |
| coverage_fraction | Fraction of parcel area covered by available grid cells |
| mean_imputed_months | Area-weighted missing-support-month count before interpolation |

`sum_abs_delta` is not the absolute annual difference: opposite monthly errors
can cancel in `annual_delta`, but never cancel in `sum_abs_delta`. Any missing
month makes the annual error NULL; partial sums are not reported as annual scores.

Also save a CSV copy, parcel/support area mappings, monthly satellite feature
cache and QA audit, fold assignments, alpha selections, coefficients/scalers,
reference predictions and metrics. Metrics include monthly MAE/RMSE, annual MAE
and bias, seasonal cosine similarity and a paired 5,000-draw parcel bootstrap
interval for MAE improvement over the mean reference. Bootstrap whole parcels,
not individual months. This interval is conditional on the evaluated fits; it
is not a spatial-transfer or per-parcel predictive interval.

## Run and reproduce

From the project directory:

```
.venv/bin/python -m pip install -r analysis/requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -p test_seasonal.py -v
.venv/bin/python scripts/analyze_seasonal.py --config analysis/config.json
```

The script hashes source data, code, configuration and the environment lock.
Verified feature caches can be reused when rerunning the same inputs. If any
manifest entry changes, select a new output directory rather than mixing runs.
The SQLite source is always read-only and satellite reads remain bounded by
support batch size. The final GeoPackage is written to a temporary filename,
integrity-checked, then moved into place.
