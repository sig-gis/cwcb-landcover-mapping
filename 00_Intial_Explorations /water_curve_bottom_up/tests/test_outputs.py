import csv, json, unittest
from pathlib import Path

import numpy as np
from osgeo import ogr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

class TestRebuildOutputs(unittest.TestCase):
    def test_raw_target_and_no_proxy(self):
        registry=json.loads((OUT/"experiment_registry.json").read_text())
        self.assertEqual(len(registry["target"]),12)
        self.assertEqual(registry["target"][0],"cow_jan_21")
        self.assertEqual(registry["target"][-1],"cow_dec_21")
        self.assertNotIn("cow_outdoor",registry["target"])
        self.assertNotIn("cow_indoor",registry["target"])

    def test_aggregation_invariants(self):
        registry=json.loads((OUT/"experiment_registry.json").read_text())
        for value in registry["aggregation_invariant"].values():
            self.assertLessEqual(value,1e-8)

    def test_complete_object_coverage(self):
        audit=json.loads((OUT/"data_audit.json").read_text())
        expected=audit["counts"]["complete_objects"]
        for name in ("class_area_nnls","landsat_class_month_nnls"):
            with (OUT/f"complete_objects_{name}.csv").open(newline="",encoding="utf-8") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),expected)
            self.assertEqual(len({r["sp_id"] for r in rows}),expected)
            self.assertTrue(all(f"unit_{m:02d}" in rows[0] for m in range(1,13)))
            self.assertTrue(all(f"object_{m:02d}" in rows[0] for m in range(1,13)))

    def test_parcel_model_rows(self):
        with (OUT/"parcel_predictions.csv").open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),38*4)
        self.assertEqual(len({r["model_id"] for r in rows}),4)
        for r in rows:
            obs=sum(float(r[f"cow_{m}_21"]) for m in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
            self.assertAlmostEqual(obs,float(r["obs_annual"]),places=9)

    def test_spatial_row_counts(self):
        expected=json.loads((OUT/"data_audit.json").read_text())["counts"]["complete_objects"]
        ds=ogr.Open(str(OUT/"complete_objects_class_area_nnls.gpkg"),0)
        self.assertEqual(ds.GetLayer(0).GetFeatureCount(),expected)
        ds=None
        ds=ogr.Open(str(OUT/"parcel_predictions.gpkg"),0)
        self.assertEqual(ds.GetLayer(0).GetFeatureCount(),38*4)

        ds=None
        ds=ogr.Open(str(OUT/"complete_objects_seasonally_regularized.gpkg"),0)
        self.assertEqual(ds.GetLayer(0).GetFeatureCount(),expected)
        ds=None
        ds=ogr.Open(str(OUT/"parcel_predictions_seasonally_regularized.gpkg"),0)
        self.assertEqual(ds.GetLayer(0).GetFeatureCount(),38)

    def test_refinement_is_bottom_up_and_improves_point_metrics(self):
        result=json.loads((OUT/"refinement_results.json").read_text())
        self.assertLessEqual(result["seasonal_aggregation_max_abs_kgal"],1e-8)
        self.assertLessEqual(result["annual_shape_aggregation_max_abs_kgal"],1e-8)
        metrics={r["model_id"]:r for r in result["metrics"]}
        refined=metrics["seasonally_regularized_class_area"]
        simple=metrics["class_area_nnls"]
        self.assertLess(refined["monthly_mae_kgal"],simple["monthly_mae_kgal"])
        self.assertGreater(refined["mean_cosine_similarity"],simple["mean_cosine_similarity"])

    def test_manifest_hashes(self):
        import hashlib
        manifest=json.loads((OUT/"output_manifest.json").read_text())
        for entry in manifest["outputs"]:
            p=ROOT/entry["path"]
            self.assertTrue(p.exists())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),entry["sha256"])

if __name__=="__main__": unittest.main()
