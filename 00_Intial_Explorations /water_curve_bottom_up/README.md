# Bottom-up monthly water-use experiments

This folder contains the clean-room 2021 parcel water-use analysis developed
from three authoritative inputs. It estimates raw January–December metered use
from land-cover object areas, compares continuous Landsat conditioning, tests
seasonal regularization, and separately examines signed land-cover effects.

## Main findings

- The simple nonnegative class-area model modestly improves on a mean curve.
- A seasonally regularized class-area model has the best point metrics, but its
  paired improvement is small and not decisive with 38 parcels.
- Continuous Landsat slightly improves average curve shape but not magnitude.
- Allowing signed class effects changes the allocation and slightly improves
  annual MAE, but does not materially improve overall held-out prediction.
- Landsat-only models recover the broad seasonal shape but perform approximately
  like the mean-curve baseline and trail the strongest land-cover model.

Object values are latent model attributions, not directly observed irrigation.

## Reproduce in Colab

Open `water_curve_bottom_up_colab.ipynb`. Set `DRIVE_INPUT_DIR` to the Google
Drive folder containing these exact filenames:

- `parcels_water_2021.gpkg`
- `landcover_2021.gpkg`
- `satellite_timeseries.sqlite`

The notebook mounts Drive, clones this repository if necessary, stages those
three files in the Colab runtime, installs dependencies, and runs:

1. `src.pipeline`
2. `src.refinement`
3. `src.signed_effects`
4. `src.landsat_only`
5. `src.build_uncertainty_report`
6. `src.export_model_artifacts`
7. `src.build_technical_report`
8. the test suite

No archived predictions, parameters, crosswalks, or derived targets are used.

## Contents

- `src/`: spatial joining, modeling, validation, uncertainty, and report code.
- `tests/`: target, aggregation, coverage, spatial-output, and manifest tests.
- `reports/water_use_uncertainty_report.html`: six-method uncertainty report.
- `reports/signed_landcover_effects_report.html`: separate signed-effects report.
- `reports/landsat_only_report.html`: four class-free Landsat-only experiments.
- `reports/TECHNICAL_METHODS_AND_FIVE_YEAR_HANDOFF.md`: authoritative method,
  theory, validation, uncertainty, model-selection, and scale-up specification;
  a standalone HTML rendering is included beside it.
- `artifacts/`: fitted coefficients, feature and class ordering, serialized
  forests, reconstruction checks, provenance, and SHA-256 manifest.
- `outputs/`: compact audit, metrics, paired comparisons, and class-rate tables.
- `inputs/README.md`: input schema and placement instructions; source data are
  intentionally excluded from Git.

Water units are thousand gallons per month. Areas are measured in square metres
in EPSG:32613; unit rates are reported per 100 square metres.

## Using the five-year database

Treat the included 2021 artifacts as a verified reference fit, not as
coefficients that can automatically be transferred to a new population. The
technical handoff specifies the required long-form parcel-month, object, and
pixel-month tables; grouped parcel/year validation; QA and Landsat scaling;
bottom-up aggregation invariants; and the refit-versus-transfer checks needed
for the larger water-use and Landsat panel.
