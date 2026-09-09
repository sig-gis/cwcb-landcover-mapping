import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import polars as pl
from unittest.mock import patch
from scripts import load_landsat as loader


class LoaderTest(unittest.TestCase):
    def test_invalid_identity_and_geometry(self):
        valid = {'system:index': ['LC08_033032_20210103_0'], 'SR_B1': [123], '.geo': [json.dumps({'type': 'Point', 'coordinates': [-105.1, 39.9]})]}
        for column, value in [('system:index', 'LC09_033032_20210103_0'), ('system:index', None), ('.geo', json.dumps({'type': 'Point', 'coordinates': [181, 39.9]})), ('.geo', None)]:
            frame = pl.DataFrame({**valid, column: [value]}, schema_overrides={'system:index': pl.String, '.geo': pl.String})
            with self.assertRaises(ValueError):
                loader.clean(frame, ['SR_B1'], 8)

    def test_batches_dedup_schema_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'csv'
            source.mkdir()
            geo = json.dumps({'type': 'Point', 'coordinates': [-105.1, 39.9]})
            def write(name, header, rows):
                with (source/name).open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
            header = ['system:index','SR_B1','.geo']
            write('Landsat8_a.csv', header, [['LC08_033032_20210103_0',123,geo], ['LC08_033032_20210103_1',123,geo], ['LC08_033032_20210104_0',456,geo]])
            write('Landsat8_b.csv', header, [['LC08_033032_20210103_99',123,geo], ['LC08_033032_20210104_1',457,geo]])
            write('Landsat9_a.csv', ['system:index','QA_RADSAT','.geo'], [['LC09_033032_20211107_0',0,geo]])
            write('Landsat9_b.csv', ['system:index','QA_RADSAT','SR_B1','.geo'], [['LC09_033032_20220114_0',0,999,geo]])
            (source/'Landsat9_empty.csv').write_text('\n\n')
            db = root/'test.sqlite'
            original = loader.clean
            calls = 0
            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interruption')
                return original(*args)
            with patch.object(loader, 'clean', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    loader.build(source, db, 1)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute('SELECT SUM(rows_loaded) FROM _load_files').fetchone()[0], 1)
            report = loader.build(source, db, 1)
            self.assertEqual(report['tables']['landsat_8']['rows'], 3)
            self.assertEqual(report['tables']['landsat_8']['duplicates_removed'], 2)
            with sqlite3.connect(db) as conn:
                self.assertIsNone(conn.execute("SELECT SR_B1 FROM landsat_9 WHERE date='2021-11-07'").fetchone()[0])
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM landsat_8_unique_locs').fetchone()[0], 1)
                plan = conn.execute('EXPLAIN QUERY PLAN SELECT * FROM landsat_8 WHERE unique_loc_id=1 ORDER BY date').fetchall()
                self.assertIn('idx_landsat_8_loc_date', str(plan))
            self.assertEqual(loader.build(source, db, 2), report)
            (source/'Landsat9_empty.csv').write_text('\n')
            with self.assertRaisesRegex(ValueError, 'Source files changed'):
                loader.build(source, db, 1)


if __name__ == '__main__':
    unittest.main()
