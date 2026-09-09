from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'outputs/seasonal_2021_size_binned'
new=pd.read_csv(out/'parcel_predictions.csv');old=pd.read_csv(root/'outputs/seasonal_2021/parcel_predictions.csv');m=json.loads((out/'metrics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True,layout='constrained')
for ax,frame,label in zip(axes,[old,new],['Original shared fit','Separate fits by parcel size']):
 frame=frame[frame.status=='eligible'];score=(frame.observed_total-frame.predicted_total)/(frame.observed_total+frame.predicted_total)
 ax.hist(score,bins=np.linspace(-1,1,81),color='#487b9e',edgecolor='white',linewidth=.3)
 ax.axvline(0,color='black',lw=1);ax.axvline(score.mean(),color='#c55712',lw=1.5,label=f'Mean {score.mean():+.3f}')
 ax.set(title=label,xlabel='(observed − predicted) / (observed + predicted)',xlim=(-1,1));ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Parcels');fig.savefig(out/'departure_histogram_comparison.png',dpi=170)
rows=['| Size class (m²) | Parcels | Underpredicted | Monthly MAE (kgal) | Annual bias per parcel (kgal) |','| --- | ---: | ---: | ---: | ---: |']
labels=['<500','500–1,000','1,000–2,000','2,000–10,000','10,000–50,000','≥50,000']
for label,g in zip(labels,m['by_size_class']):
 b=g['binned'];rows.append(f"| {label} | {g['parcels']:,} | {b['underpredicted_fraction']:.1%} | {b['monthly_mae_kgal']:.3f} | {b['annual_bias_kgal']:.3f} |")
text='''# Size-binned analysis completed

Primary deliverable: `Westminster_2021_size_binned.gpkg`, layer `parcel_water_use`.
30,912 parcel geometries are retained; 30,274 have held-out predictions. The
remaining 638 retain their version 1 exclusion status and NULL predictions.
Source observations, geometry and outer-fold assignments are unchanged.

Six independent size-class models use the same nonnegative Landsat equation,
indices, area weights and alpha grid. Bin boundaries were fixed before fitting.
Details: `../../analysis/size_binned/METHODS.md`.

| Overall measure | Original | Size-binned |
| --- | ---: | ---: |
| Parcels with observed annual use above prediction | 88.5% | 44.9% |
| Aggregate annual volume underprediction | 50.8% | 4.3% |
| Monthly MAE, kgal | 7.719 | 6.807 |
| Monthly RMSE, kgal | 51.012 | 48.973 |
| Mean relative departure | +0.332 | -0.049 |
| Median relative departure | +0.382 | -0.028 |

Annual observed volume is 3,708,785 kgal; new predicted volume is 3,550,300 kgal.
The fraction underpredicted and total volume bias need not have matching signs:
parcels differ greatly in consumption. No offset or post-hoc recentering was used.

## Results by class

'''+ '\n'.join(rows)+'''

The main-size classes now have substantially more balanced departures. Larger
classes still have imbalanced signs and sparse data; this is not a guarantee
that each class supplies an unbiased expectation. The largest class has only
63 eligible parcels. Class boundaries can introduce discontinuities.

The class-specific mean-curve reference has monthly MAE 7.132 kgal versus 6.807
for the binned Landsat model, but lower RMSE (46.664 versus 48.973). These are
mixed metrics, not uniform superiority. The revised design was motivated by
version 1 results and reuses its folds; it is not a newly untouched test set.

## QGIS fields

- `observed_01`–`observed_12`, `predicted_01`–`predicted_12`: January–December kgal.
- `delta_01`–`delta_12`: predicted minus observed, kgal.
- `sum_abs_delta`: sum of twelve absolute monthly errors, kgal.
- `size_class_id`, `size_class`: assigned size class, IDs 0–5.
- `signed_departure`: observed annual total minus predicted annual total, kgal.
- `relative_departure`: (observed−predicted)/(observed+predicted), matching the
  user's expression after cancellation of area. Positive means observed exceeds
  predicted. Missing inputs or a zero denominator yield NULL.

`annual_delta` keeps the original opposite sign: predicted minus observed.

## Validation and reproduction

Tests verify bin boundaries and that changing held-out target values cannot
change their model's predictions or selected regularization. The run checks
SQLite integrity/foreign keys, exact saved geometry and monthly values, error
calculations and all held-out predictions reconstructed from saved class/fold
model states. Maximum reconstruction discrepancy: 1.19e-11 kgal.

Code/config/environment and reused input hashes are in `manifest.json`;
`model_state.npz`, `model_selection.json`, `training_counts.csv`,
`fold_assignments.csv`, `metrics.json`, and `validation.json` preserve decisions
and results. The original version has not been overwritten.
'''
(out/'RESULTS.md').write_text(text)
print('\n'.join(rows))
