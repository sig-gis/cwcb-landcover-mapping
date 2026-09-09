# Neighborhood field sampling

We selected complete street blocks and cul-de-sacs to provide field examples of four patterns in observed water use relative to predictions from the existing parcel-size-binned Landsat model. Each circuit includes the relevant corner properties and a nearby park, greenbelt or common space. Circuits were individually reviewed for a coherent walk, complete parcel coverage and limited street crossings. The selected parcels, street legs, paired spaces and suggested starts are recorded in the review inputs.

For each modeled parcel, we calculate annual relative departure as:

`(observed_total - predicted_total) / (observed_total + predicted_total)`

We also calculate monthly disagreement: the sum of the 12 absolute monthly differences divided by combined annual observed and predicted use. Each circuit requires at least four modeled assessment parcels, at least 80% model coverage and modeled parcels on both reviewed sides of the street. Context parcels do not contribute to these statistics.

| Category | Definition among modeled assessment parcels |
|---|---|
| Above prediction | At least 80% positive departure and at least 50% at or above +0.20 |
| Below prediction | At least 80% negative departure and at least 50% at or below −0.20 |
| Mixed | At least 25% at or above +0.20 and at least 25% at or below −0.20 |
| Close to prediction | At least 60% within ±0.10, also with monthly disagreement no greater than 0.20 |

Within each category, rank first by the recorded crossing-exposure priority, then by category fit, then by the number of modeled parcels. Category fit is the fraction at or above +0.20 for the above group, at or below −0.20 for the below group, and meeting both close criteria for the close group. For mixed circuits it is twice the smaller of the above and below fractions, favoring balanced examples. Circuit ID resolves exact ties. The first three circuits per category are primary choices; the remainder are backups.

The reviewed selection contains 53 circuits and 657 assessment parcels: 12 above prediction, 14 below, 14 mixed and 13 close. This purposive sample provides contrasting examples; it is not a random sample of Westminster. Departure indicates disagreement with the model, not demonstrated overwatering or underwatering. Suggested starts and street-centerline routes are planning references; parking and entrances require field confirmation.

The reviewed selection is an explicit input. Reproduction means rebuilding its parcel geometry, walks, categories, rankings and maps from the recorded decisions and source data. Applying the design to another area requires a new site review.

From the project folder, run:

```bash
MPLCONFIGDIR=/tmp/cwcb-matplotlib .venv/bin/python scripts/field_sampling.py --output outputs/my_field_sample --verify
```

Use a new output directory. The command generates the GeoPackage, circuit summary, detailed PDF maps, `top_ranked_streets.png` and `all_sites_overview.png`. It then builds everything again in an empty temporary directory and compares all results. `reproduction_check.json` records the comparison; GeoPackage creation timestamps are excluded.
