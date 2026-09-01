# Bottom-up 2021 monthly parcel water-use analysis

## Answer

Land-cover composition improves on the training-mean curve in LOPO monthly MAE. Continuous Landsat does not deliver a consistent material improvement over the simple class model. The preferred balance is **class_area_nnls** by raw held-out MAE, but complexity and paired uncertainty are considered below; object curves are latent attributions, not measured irrigation or observed object use.

## Data and support audit

The analysis uses exactly 38 parcels, 13,019 complete aerial objects, and 10,847 positive-area parcel/object fragments across 13 classes. Areas are computed in EPSG:32613. The response is the unaltered 12 monthly `cow_*_21` fields in thousand gallons/month; neither `cow_outdoor` nor `cow_indoor` is used. There are 0 missing response values.

Landsat 8/9 Collection 2 surface reflectance is scaled by `DN × 0.0000275 - 0.2`. QA_PIXEL bits 0–5 and nonzero QA_RADSAT are rejected. Clear observations become monthly medians of NDVI, NDMI, and EVI; 0 of 1800 pixel-months lack a clear acquisition and are linearly interpolated with endpoint carry. Complete aerial-object centroids are assigned to the nearest one of 150 Landsat support points; 60 are used. Multiple aerial objects share these ~30 m contextual pixels—the satellite series does not directly resolve them. Full schemas, null counts, table date ranges, class counts/areas, and pixel-sharing counts are in `outputs/data_audit.json`.

## Formulations

1. **Mean curve:** each held-out parcel receives the 12-month mean of the other 37 parcels.
2. **Class-area NNLS:** each class has one nonnegative 12-month unit curve in kgal/100 m²/month. A fragment contribution is its area/100 times that class rate; parcel predictions are exact sums. Ridge strength is selected within training data.
3. **Proportional multi-output forest:** one 700-tree forest maps class proportions to all 12 months. It is retained honestly as a parcel-only benchmark. It has no defensible object unit-rate allocation.
4. **Continuous Landsat class-month NNLS:** each class-month rate is a nonnegative linear intercept plus nonnegative effects of continuous, robust-scaled same-month NDVI, NDMI, and EVI. Scaling and regularization selection are inside outer training folds. This produces object rates before aggregation.

## Held-out results

| model | monthly MAE | monthly RMSE | mean cosine | annual MAE | annual bias |
|---|---:|---:|---:|---:|---:|
| mean_curve | 3.999 | 7.522 | 0.919 | 38.831 | 0.000 |
| class_area_nnls | 3.714 | 7.370 | 0.914 | 34.953 | -2.534 |
| proportion_multioutput_rf | 4.042 | 7.432 | 0.920 | 39.319 | 2.817 |
| landsat_class_month_nnls | 3.831 | 7.412 | 0.922 | 36.486 | 0.877 |

All values other than cosine are kgal (monthly values per parcel-month; annual values per parcel-year). Month-level error/bias is in `monthly_diagnostics.csv`; every held-out parcel curve is shown below and tabulated in the parcel outputs.

![Held-out parcel curves](held_out_parcel_curves.png)

The simple model changes monthly MAE versus the mean baseline by -0.285 kgal. The forest changes it versus simple by +0.327; Landsat changes it by +0.116. Paired parcel bootstrap intervals and win counts are in `paired_method_differences.csv`; tiny differences whose interval spans zero are not treated as meaningful.

## Object consequences and stability

`class_unit_curves.csv` provides the directly interpretable simple-model rates. Complete-object GeoPackages contain each original, unclipped object, its shared Landsat support, monthly unit and whole-object curves, annual sums, support flags, and joint parcel-bootstrap intervals for retained finalists. These intervals come from common model refits, so shared parameters and shared pixels remain dependent. The same draws are aggregated to parcels; independent object intervals are never summed.

The maximum numerical discrepancy between direct parcel predictions and summed fragment contributions is 2.84e-14 kgal. Large/bootstrap-wide or fold-variable rates should not be transferred casually: only 38 parcel totals identify 13 latent class curves, and correlated class areas limit attribution stability.

## Forest counterfactuals

For each class, its proportion is set to zero while all other proportions remain unchanged; the vector is deliberately **not renormalized**, so its sum decreases. Monthly and annual changes are in `forest_class_zeroing.csv`. These are model counterfactuals, not causal effects and not object-level unit rates.

## Complexity decision

The preferred method is the simplest model whose held-out gains are practically supported. The satellite model is retained as a finalist only when its overall LOPO monthly MAE beats the simple model; it is described as materially better only when it also improves annual MAE and paired uncertainty supports the gain. The forest is not promoted to bottom-up status because forcing an allocation would add an unidentifiable rule.

## Archive audit

Only three archived ideas were inspected. The old additive code used raw monthly targets but included a parcel intercept, so it was reimplemented here without that non-object term. The old proportional forest did predict 12 raw monthly values but remained parcel-level; only its transparent zeroing convention was reused. The old continuous Landsat model predicted scalar `cow_outdoor`, not the required monthly curve; its scores are discarded, while its QA/scaling and nearest-pixel mechanics were independently rebuilt from raw inputs. Curve grouping, two-regime models, extra trees, and unrelated archived experiments are out of scope.

## Limitations

These are predictive latent attributions from 38 parcels, not causal water demands. Parcel coverage may omit slivers or include overlaps; totals are documented in the audit. Landsat interpolation is contextual and shared. Nonnegativity improves coherence but can pin weakly supported effects at zero. Bootstrap ranges reflect parcel sampling instability, not all measurement or land-cover classification uncertainty.
