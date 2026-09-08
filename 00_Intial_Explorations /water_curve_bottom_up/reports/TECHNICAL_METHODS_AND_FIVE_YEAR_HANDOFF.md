# Technical methods and five-year implementation handoff

## 1. Purpose and scope

This document is the authoritative technical handoff for the bottom-up monthly
water-use experiments. It explains the estimand, spatial construction,
satellite processing, every model tested, validation and uncertainty, retained
methods, fitted artifacts, and the recommended transition to a larger five-year
Landsat 8/9 and parcel water-use database.

The fitted 2021 artifacts are supplied to reproduce the completed experiment.
They are not assumed to transfer unchanged to a larger population. The larger
dataset should be used to retrain and independently validate the retained model
families.

## 2. Estimand

The modeled response is the unmodified 12-element parcel vector

`y[p] = (cow_jan_21, ..., cow_dec_21)`

in thousand gallons per month. Annual volume is `sum_m y[p,m]`. No experiment
subtracted `cow_indoor`, used `cow_outdoor` as its target, normalized the curve
to another total, or constructed an irrigation proxy. Therefore all results are
estimates or attributions of **total metered water use**, not measured outdoor
use.

For a bottom-up model, object `i` in parcel `p` has a monthly unit rate
`r[i,m]` in kgal per 100 m² per month. Its contribution is

`w[i,m] = clipped_area[i,p] / 100 × r[i,m]`.

The parcel prediction is

`ŷ[p,m] = Σ_i∈p w[i,m]`.

Complete-object deliverables evaluate the same unit-rate equation on each
original unclipped aerial object. Object values are latent model attributions,
not directly observed consumption.

## 3. Data and spatial support

The study contains 38 parcel polygons, 13,019 complete aerial land-cover
objects, 10,847 positive-area object–parcel fragments, 13 land-cover classes,
and 150 available Landsat support locations. Sixty Landsat supports are used by
the complete aerial objects.

Parcel geometries are WGS84 and land-cover objects are Web Mercator. All areas
are calculated after transformation to EPSG:32613. Aerial objects are intersected
with parcels, and intersection areas—not full-object areas—are used for parcel
aggregation. Complete-object outputs retain original geometry and area.

Each complete aerial object is associated with the nearest Landsat support to
its centroid. Every fragment inherits its complete object's support. Multiple
small aerial objects share a Landsat pixel. Landsat is consequently contextual
~30 m information; it does not directly resolve each aerial object.

## 4. Landsat preprocessing

Landsat 8 and 9 Collection 2 surface-reflectance observations from 2021 are
combined. Surface-reflectance digital numbers use

`reflectance = DN × 0.0000275 - 0.2`.

Observations are rejected when any QA_PIXEL bit 0–5 is set (fill, dilated cloud,
cirrus, cloud, cloud shadow, or snow) or QA_RADSAT is nonzero. Three continuous
indices are calculated:

- `NDVI = (NIR - red) / (NIR + red)`
- `NDMI = (NIR - SWIR1) / (NIR + SWIR1)`
- `EVI = 2.5 × (NIR - red) / (NIR + 6 red - 7.5 blue + 1)`

Clear observations are reduced to a median for each pixel, month, and index.
Missing pixel-month values are linearly interpolated through time with endpoint
carry. Models use the complete continuous series, never two satellite regimes.

Bottom-up satellite variables are robust-scaled within training parcels:

`z = clip((x - training p05) / (training p95 - training p05), 0, 1)`.

Parcel ridge uses standard scores learned from training parcels. Random forests
use unscaled features. All outer-fold preprocessing excludes the held-out parcel.

## 5. Validation and evaluation

The primary validation is leave-one-parcel-out (LOPO). For each of 38 folds,
one complete parcel is excluded, all fitting and learned preprocessing use the
remaining 37 parcels, and a 12-month prediction is produced for the excluded
parcel. Hyperparameters are selected only within outer-training parcels.

Reported measures are:

- monthly MAE and RMSE across 456 held-out parcel-month values;
- month-specific error and bias;
- cosine similarity of each observed and predicted 12-month curve;
- MAE, RMSE, and bias of annual sums;
- paired parcel-level MAE differences with parcel bootstrap intervals.

The uncertainty reports use two distinct quantities. Aggregate-curve bands are
paired parcel-bootstrap confidence intervals for the mean held-out prediction.
Example-parcel bands are leave-one-out residual-calibrated predictive-error
bands. Finalist parameter uncertainty uses 300 joint parcel-bootstrap refits.
The same coefficient draw is applied to all objects and then aggregated, so
shared coefficients and shared Landsat supports remain dependent. Independent
object intervals are never summed.

## 6. Methods tested

### 6.1 Training-mean curve

The held-out prediction is the mean of the other 37 response curves. This tests
whether any feature model improves on the common seasonal cycle. It is a
parcel-level baseline and has no object attribution.

### 6.2 Nonnegative class-area model

For class `c` and month `m`, estimate `β[c,m] ≥ 0`:

`ŷ[p,m] = Σ_c area[p,c]/100 × β[c,m]`.

Coefficients minimize squared parcel error plus ridge shrinkage. Ridge strength
is selected inside validation. This is the simplest directly interpretable
bottom-up model: every complete object receives its class curve.

### 6.3 Proportional multi-output random forest

One 700-tree random forest maps 13 parcel class proportions to all 12 months
simultaneously. It uses bootstrap trees, minimum leaf size 2, and 70% feature
subsampling. It is a nonlinear parcel benchmark, not an object model. Class
zeroing leaves other proportions unchanged and does not renormalize the vector;
those changes are predictive counterfactuals, not causal effects or unit rates.

### 6.4 Class-specific Landsat-conditioned nonnegative response

For each object class and month:

`r[i,m] = β0[class_i,m] + β[class_i,m] · z[i,m]`, with all coefficients nonnegative.

Here `z` contains monthly NDVI, NDMI, and EVI. Rates are multiplied by object
area and summed. This tests whether satellite condition improves a known-class
unit curve. It increased complexity without materially improving held-out
magnitude.

### 6.5 Annual class-area magnitude with common shape

First estimate a nonnegative annual class-area rate. Then distribute every
object's annual attribution using the aggregate monthly shape learned from the
training parcels. This responds to the observation that the mean curve already
captures shape well. It slightly improved annual MAE but gave up monthly MAE.

### 6.6 Seasonally regularized nonnegative class-area model

All `13 × 12` class rates are fitted jointly. The objective contains parcel
squared error, ridge shrinkage, adjacent-month first-difference smoothing, and
an optionally tuned class-pooling penalty, subject to `β ≥ 0`.

The full-data selected configuration is ridge 10, smoothness 10, pooling 0.
Temporal smoothing improved every headline point metric relative to the simple
class model, but the paired improvement interval includes zero. This is the
best point-metric model and the preferred operational candidate when its modest
additional complexity is acceptable.

### 6.7 Signed ridge class-area model

This is the direct ridge counterpart of the simple class-area model, without
the nonnegativity constraint. A negative class-month rate means that class
reduces predicted metered-tap requirement within the fitted additive accounting.
The final meter prediction need not be negative, and none of the full-fit parcel
months was negative.

### 6.8 Signed seasonally regularized class-area model

This combines signed class rates with ridge and adjacent-month smoothing. It is
the retained signed sensitivity model. Signed effects changed attribution and
slightly improved annual MAE, but did not materially improve overall held-out
prediction. No class had a 90% annual bootstrap interval wholly below zero.

### 6.9 Landsat-only parcel ridge

Fragment-area-weighted monthly NDVI, NDMI, and EVI are summarized to a
36-element parcel trajectory. A standardized multi-output ridge maps that
trajectory to 12 water-use values. No land-cover class or proportion enters.
It performed approximately like the mean curve.

### 6.10 Landsat-only parcel random forest

The same 36 parcel Landsat variables feed one 700-tree multi-output forest. It
had the best Landsat-only cosine similarity by a negligible margin but the worst
overall magnitude errors. It showed a potentially useful ability to recognize
some high-use parcels, so it is retained as a nonlinear upper-tail benchmark,
not as the preferred average-error model.

### 6.11 Landsat-only bottom-up nonnegative response

Class identity is removed. Each object's monthly unit rate is

`r[i,m] = β0[m] + β[m] · z[i,m]`, with nonnegative coefficients.

Area-weighted object contributions sum to parcels. This was the best
Landsat-only magnitude model, but was statistically indistinguishable from the
mean curve and worse than the land-cover seasonal model.

### 6.12 Landsat-only bottom-up signed response

This uses the same equation without nonnegative coefficient constraints.
Allowing signed satellite effects did not improve held-out performance. It is
retained only as a sensitivity result.

## 7. Comparative results

Headline LOPO results are reproduced in the machine-readable CSV files. The
central comparison is:

| Model | Monthly MAE | Monthly RMSE | Mean cosine | Annual MAE |
|---|---:|---:|---:|---:|
| Mean curve | 3.999 | 7.522 | 0.919 | 38.831 |
| Simple class-area | 3.714 | 7.370 | 0.914 | 34.953 |
| Seasonal class-area | **3.667** | **7.342** | 0.920 | 34.733 |
| Signed seasonal class-area | 3.665 | 7.394 | 0.905 | **34.586** |
| Landsat-only parcel ridge | 3.995 | 7.606 | 0.921 | 39.553 |
| Landsat-only parcel forest | 4.290 | 8.014 | **0.921** | 43.061 |
| Landsat-only bottom-up nonnegative | 3.940 | 7.643 | 0.920 | 38.428 |

Small numerical differences are not automatically meaningful. The seasonal
nonnegative model improved monthly MAE over the simple class model by only 0.047
kgal, with a paired interval spanning zero. Signed seasonal and nonnegative
seasonal monthly MAE differed by 0.003 kgal. Landsat-only models did not clearly
beat the mean curve.

## 8. Retained methods and intended roles

1. **Seasonally regularized nonnegative class-area:** preferred point-metric,
   bottom-up operational candidate.
2. **Simple nonnegative class-area:** preferred transparent reference and
   fallback when maximum simplicity is important.
3. **Signed seasonally regularized class-area:** sensitivity analysis for real
   land-cover reductions in required metered water.
4. **Landsat-only bottom-up nonnegative:** best satellite-only magnitude
   benchmark and a portable object-to-parcel construction.
5. **Landsat-only parcel forest:** exploratory nonlinear and high-use/upper-tail
   benchmark; not preferred by average error.
6. **Mean curve:** mandatory baseline.

The remaining methods are documented negative or intermediate results. None of
the retained methods has been validated as an outdoor-water estimator because
the target is total metered use.

## 9. Portable fitted artifacts

`artifacts/linear_model_state.npz` contains all linear coefficients, class and
target order, common shape, Landsat scalers, and parcel-ridge state.
`artifacts/class_month_coefficients.csv` is a human-readable extract of the four
class-area coefficient matrices. Forests are stored as compressed joblib files.

`artifacts/artifact_definitions.json` defines every array, equation, role, unit,
and feature order. `artifacts/artifact_manifest.json` provides SHA-256 hashes for
artifacts and source inputs. `artifacts/reconstruction_checks.json` verifies
that retained coefficient matrices exactly reproduce saved full-fit parcel
predictions.

The NPZ archive is preferred over pickle for linear models because it is
language-neutral at the array level. Joblib forests require compatible Python,
NumPy, and scikit-learn versions. For long-term use, retraining forests from
documented settings is safer than depending on indefinite pickle compatibility.

## 10. Moving to a five-year dataset

### 10.1 Required analytical table

Construct one row per parcel-year and retain a separate object-year-month table.
At minimum:

- stable `parcel_id`, calendar `year`, and 12 raw monthly meter values;
- parcel geometry and area;
- stable `object_id`, class, complete geometry, parcel-clipped area, and year;
- Landsat support ID and coordinates;
- observation date, sensor, scaled bands, QA fields, and monthly indices;
- counts of clear acquisitions and imputation flags for every pixel-month.

Preserve the distinction between a complete object, its parcel fragment, and a
shared Landsat pixel. If land cover changes by year, version objects/classes and
areas rather than silently applying 2021 cover to every year.

### 10.2 Do not apply the 2021 coefficients unchanged

The existing coefficients encode one small geography and one year. Five-year
data introduce sensor sampling differences, climate variation, changing cover,
meter/account changes, and repeated observations from the same parcels. Use the
artifacts to verify feature engineering and reproduce the 2021 benchmark, then
retrain on the larger panel.

### 10.3 Recommended model sequence

1. Reproduce the mean, simple class-area, seasonal class-area, Landsat-only
   bottom-up, and parcel forest baselines on the expanded data.
2. Fit a panel extension of the seasonal class model:
   `class seasonal curve + year/weather response + satellite anomaly response`.
3. Express satellite variables as both absolute monthly values and within-pixel
   anomalies from that pixel's multi-year monthly normal.
4. Add current and antecedent precipitation or reference ET when available;
   otherwise Landsat may proxy weather inconsistently.
5. Permit signed satellite and class deviations while monitoring final parcel
   predictions and attribution stability.
6. Add complexity only when grouped held-out results identify a specific gain.

### 10.4 Validation for repeated parcels and years

Ordinary random parcel-month splits would leak parcel identity and adjacent-time
information. Use at least two assessments:

- **Unseen parcels:** grouped spatial cross-validation holding out every year of
  a parcel, preferably with geographic blocking to limit neighborhood leakage.
- **Unseen years:** forward or leave-one-year-out validation to measure temporal
  transfer under different weather and acquisition conditions.

A stringent deployment test holds out both parcels and the latest year. All
scaling, imputation rules that learn population values, hyperparameter tuning,
feature selection, and dimensional reduction must be learned inside the
training partition.

Report overall metrics plus strata for annual-use quantiles, parcel size,
Landsat clear-observation coverage, class support, year, and geography. Because
the forest appeared more responsive to some high-use parcels, report upper-tail
annual MAE, bias, recall of high-use parcels, and interval calibration rather
than relying only on global MAE.

### 10.5 Five-year bottom-up formulation

A practical retained extension is

`r[i,y,m] = positive_link(class seasonal baseline + class × Landsat anomaly + weather effects)`

or a signed-deviation form

`r[i,y,m] = shared positive baseline + signed class deviation + signed satellite deviation`.

Then

`ŷ[p,y,m] = Σ_i∈p clipped_area[i,p,y]/100 × r[i,y,m]`.

Partial pooling should operate across classes, months, years, and possibly
neighborhoods. It should not erase real year-to-year variation. With many
parcels, hierarchical or penalized estimation can stabilize rare classes while
allowing common seasonal and weather responses.

### 10.6 Uncertainty and calibration

Use grouped parcel bootstrap or hierarchical posterior draws, resampling entire
parcel histories rather than individual months. Apply every draw jointly to all
objects and aggregate the same draws to parcels. Evaluate monthly, annual, and
upper-tail interval coverage on unseen parcels and years. For forests, use
out-of-fold conformal or quantile calibration; raw variation among trees is not
a complete predictive interval.

### 10.7 Outdoor-use interpretation

Scaling the dataset does not by itself change the target from total to outdoor
water. If outdoor use is the desired estimand, obtain irrigation meters,
high-frequency disaggregation, audits, or a jointly estimated base-use component.
A multi-year joint model can use winter/base behavior without subtracting a
fixed proxy, but it must be labeled as a latent decomposition and externally
validated before its landscape attribution is called outdoor consumption.

## 11. Reproduction order

With the three authoritative inputs in `inputs/`, run:

```bash
python -m src.pipeline
python -m src.refinement
python -m src.signed_effects
python -m src.landsat_only
python -m src.export_model_artifacts
python -m src.build_uncertainty_report
python -m unittest discover -s tests -v
```

Limit numerical threads as shown in the project README. The Colab notebook
mounts Google Drive, stages only the three inputs, executes this sequence, and
links the resulting reports.

## 12. Reproducibility checklist

- Verify source hashes before fitting.
- Preserve raw target fields and declared units.
- Recreate object–parcel intersections from source geometries.
- Recreate object–Landsat association from coordinates.
- Record QA masks, scale/offset, monthly reducers, observation counts, and gaps.
- Freeze class, month, and feature ordering.
- Fit preprocessing inside validation partitions.
- Save held-out predictions separately from full-fit predictions.
- Verify object-fragment sums against parcel predictions numerically.
- Save negative and failed results.
- Version code, environment, artifacts, schemas, and hashes together.
- Retrain and recalibrate before applying to a new population or year.

