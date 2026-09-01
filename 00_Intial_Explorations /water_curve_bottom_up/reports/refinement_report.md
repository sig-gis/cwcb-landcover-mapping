# Focused second-round model refinement

## Priorities

Preserve the raw monthly target and exact bottom-up aggregation; address the observed magnitude-versus-shape failure; stabilize class curves with minimal added structure; use identical LOPO splits; and reject complexity whose paired improvement is not credible.

## Models tried

**Annual class-area magnitude with common shape.** Nonnegative class-area coefficients estimate annual magnitude. The held-out annual estimate is distributed using the aggregate monthly shape learned only from outer-training parcels. Every object receives its class annual rate times that common shape.

**Seasonally regularized class-area model.** All 12 class-month rates are fitted jointly. Rates remain nonnegative, adjacent months receive a smoothness penalty, and an optional class-pooling penalty is selected inside each outer fold. The full-data selection chose ridge 10, smoothness 10, and no pooling. This is still `area × class unit curve`, not a parcel-only model.

## Results

| model | monthly MAE | RMSE | mean cosine | annual MAE |
|---|---:|---:|---:|---:|
| original class-area | 3.714 | 7.370 | 0.914 | 34.953 |
| annual magnitude/common shape | 3.729 | 7.360 | 0.919 | 34.792 |
| seasonally regularized | **3.667** | **7.342** | **0.920** | **34.733** |

The seasonally regularized model improves all four summaries, but monthly MAE improves by only 0.047 kgal (1.3%). It wins on 25 of 38 parcels. The paired bootstrap interval for original-minus-regularized MAE is [-0.013, 0.108] kgal and includes zero. Therefore this is a plausible, well-targeted improvement, not decisive evidence of superiority.

Outer folds also selected different pooling strengths, while the full fit selected none. That instability says the data support temporal smoothing more clearly than hierarchical class pooling.

## Decision

Package the seasonally regularized model as an improved challenger and co-finalist, with complete-object and parcel GeoPackages plus 300 joint parcel-bootstrap refits. Retain the original class-area model as the default when maximum simplicity is preferred. The annual common-shape model is a useful negative result: forcing one shape restores cosine similarity but does not improve monthly MAE.

The challenger aggregation discrepancy is 1.78e-14 kgal. Object values remain latent model attributions, not measured turf or irrigation use.
