#!/usr/bin/env python3
"""Batched local Landsat 8/9 CSV loader; see README.md for input and run instructions."""
import argparse
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
import time

import polars as pl

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'inputs/landsat_csvs'
DEFAULT_DB = Path(__file__).resolve().parents[1] / 'outputs/CWCB_Westminster_2026_08_19_1700.sqlite'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def manifest(source):
    result = []
    for satellite in (8, 9):
        for path in sorted(source.glob(f'Landsat{satellite}_*.csv')):
            with path.open(newline='') as stream:
                header = next(csv.reader(stream), [])
            result.append(dict(path=str(path.resolve()), size=path.stat().st_size,
                               mtime_ns=path.stat().st_mtime_ns, header=header,
                               table=f'landsat_{satellite}'))
    if not result:
        raise ValueError(f'No Landsat 8/9 CSVs in {source}')
    return result


def clean(frame, bands, satellite):
    index = pl.col('system:index')
    valid = frame.select(index.str.contains(rf'^LC0{satellite}_\d{{6}}_\d{{8}}_\d+$').fill_null(False).all()).item()
    if not valid:
        raise ValueError('Unexpected or null Landsat system:index')
    frame = frame.with_columns(
        index.str.replace(r'_[^_]+$', '').alias('image_id'),
        pl.col('.geo').str.json_decode(pl.Struct({'type': pl.String, 'coordinates': pl.List(pl.Float64)})).alias('_geo'),
    ).with_columns(
        pl.col('image_id').str.extract(r'(\d{8})$', 1).str.strptime(pl.Date, '%Y%m%d', strict=True).cast(pl.String).alias('date'),
        pl.col('_geo').struct.field('coordinates').list.get(0).alias('lon'),
        pl.col('_geo').struct.field('coordinates').list.get(1).alias('lat'),
    )
    if not frame.select(((pl.col('_geo').struct.field('type') == 'Point') &
                         (pl.col('_geo').struct.field('coordinates').list.len() == 2) &
                         pl.col('lon').is_finite() & pl.col('lat').is_finite() &
                         pl.col('lon').is_between(-180, 180) & pl.col('lat').is_between(-90, 90)).fill_null(False).all()).item():
        raise ValueError('Invalid or missing point coordinates')
    return frame.select([pl.col(b).cast(pl.Int64) if b in frame.columns else pl.lit(None, dtype=pl.Int64).alias(b)
                         for b in bands] + [pl.col(c) for c in ('image_id', 'date', 'lon', 'lat')])


def build(source=DEFAULT_SOURCE, db=DEFAULT_DB, batch_size=100_000):
    source, db = Path(source).resolve(), Path(db).resolve()
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    inputs = manifest(source)
    db.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('SQLITE_TMPDIR', str(db.parent))
    existed = db.exists()
    conn = sqlite3.connect(db)
    try:
        conn.execute('PRAGMA cache_size=-131072')
        conn.execute('PRAGMA temp_store=FILE')
        conn.execute('PRAGMA synchronous=FULL')
        if existed and not conn.execute("SELECT 1 FROM sqlite_master WHERE name='_loader_metadata'").fetchone():
            raise ValueError('Refusing to modify an existing database not created by this loader')
        conn.execute('CREATE TABLE IF NOT EXISTS _loader_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        fingerprint = json.dumps(inputs, sort_keys=True)
        previous = conn.execute("SELECT value FROM _loader_metadata WHERE key='manifest'").fetchone()
        if previous and previous[0] != fingerprint:
            raise ValueError('Source files changed since this database was started; use a new output database')
        conn.execute('INSERT OR IGNORE INTO _loader_metadata VALUES (?, ?)', ('manifest', fingerprint))
        conn.execute('CREATE TABLE IF NOT EXISTS _load_files (path TEXT PRIMARY KEY, table_group TEXT, rows_loaded INTEGER NOT NULL, complete INTEGER NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS _completed_groups (table_group TEXT PRIMARY KEY)')
        conn.execute('CREATE TABLE IF NOT EXISTS _group_stages (table_group TEXT PRIMARY KEY, stage TEXT NOT NULL)')
        conn.commit()
        for satellite in (8, 9):
            table = f'landsat_{satellite}'
            files = [f for f in inputs if f['table'] == table]
            if not files:
                continue
            if conn.execute('SELECT 1 FROM _completed_groups WHERE table_group=?', (table,)).fetchone():
                log(f'{table}: already complete')
                continue
            bands = sorted({c for f in files for c in f['header'] if c not in ('system:index', '.geo')})
            if not bands:
                log(f'{table}: no data headers, skipping')
                continue
            for f in files:
                if f['header'] and not {'system:index', '.geo'} <= set(f['header']):
                    raise ValueError(f"Missing identity/geometry columns: {f['path']}")
            columns = [b.replace(':', '_').replace('.', '_').replace(' ', '_') for b in bands] + ['image_id', 'date', 'lon', 'lat', 'unique_loc_id']
            if len(columns) != len(set(columns)):
                raise ValueError('Column names collide after sanitizing')
            definitions = [f'{quote(c)} INTEGER' for c in columns[:len(bands)]] + ['image_id TEXT NOT NULL', 'date TEXT NOT NULL', 'lon REAL NOT NULL', 'lat REAL NOT NULL', 'unique_loc_id INTEGER NOT NULL']
            raw, locs = table + '__loading', table + '_unique_locs'
            stage_row = conn.execute('SELECT stage FROM _group_stages WHERE table_group=?', (table,)).fetchone()
            stage = stage_row[0] if stage_row else 'loading'
            conn.execute(f'CREATE TABLE IF NOT EXISTS {locs} (id INTEGER PRIMARY KEY, lat REAL NOT NULL, lon REAL NOT NULL)')
            conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{locs}_latlon ON {locs}(lat, lon)')
            conn.commit()

            @lru_cache(maxsize=1_000_000)
            def location_id(lat, lon):
                found = conn.execute(f'SELECT id FROM {locs} WHERE lat=? AND lon=?', (lat, lon)).fetchone()
                if found:
                    return found[0]
                return conn.execute(f'INSERT INTO {locs}(lat,lon) VALUES (?,?)', (lat, lon)).lastrowid

            if stage == 'loading':
                conn.execute(f'CREATE TABLE IF NOT EXISTS {raw} ({", ".join(definitions)})')
                conn.commit()
                insert = f'INSERT INTO {raw} VALUES ({",".join("?" for _ in columns)})'
                for f in files:
                    path = f['path']
                    checkpoint = conn.execute('SELECT rows_loaded, complete FROM _load_files WHERE path=?', (path,)).fetchone()
                    if checkpoint and checkpoint[1]:
                        continue
                    loaded = checkpoint[0] if checkpoint else 0
                    if not f['header']:
                        if Path(path).read_text().strip():
                            raise ValueError(f'Unexpected nonempty headerless file: {path}')
                        conn.execute('INSERT OR REPLACE INTO _load_files VALUES (?,?,0,1)', (path, table))
                        conn.commit()
                        log(f'{table}: skipped empty {Path(path).name}')
                        continue
                    log(f'{table}: reading {Path(path).name}, resuming at row {loaded:,}')
                    schema = {c: pl.String if c in ('system:index', '.geo') else pl.Int64 for c in f['header']}
                    batches = pl.scan_csv(path, schema=schema).slice(loaded).collect_batches(chunk_size=batch_size, engine='streaming')
                    last_log = time.monotonic()
                    for batch in batches:
                        frame = clean(batch, bands, satellite)
                        with conn:
                            def rows():
                                for row in frame.iter_rows():
                                    yield (*row, location_id(row[-1], row[-2]))
                            conn.executemany(insert, rows())
                            loaded += len(frame)
                            conn.execute('INSERT OR REPLACE INTO _load_files VALUES (?,?,?,0)', (path, table, loaded))
                        if time.monotonic() - last_log >= 30:
                            log(f'{table}: {Path(path).name}: {loaded:,} rows committed')
                            last_log = time.monotonic()
                    conn.execute('UPDATE _load_files SET complete=1 WHERE path=?', (path,))
                    if loaded == 0:
                        conn.execute('INSERT OR REPLACE INTO _load_files VALUES (?,?,0,1)', (path, table))
                    conn.commit()
                    log(f'{table}: finished {Path(path).name}: {loaded:,} rows')
                log(f'{table}: removing exact duplicates across all files (disk-based SQL)')
                with conn:
                    conn.execute('BEGIN IMMEDIATE')
                    conn.execute(f'CREATE TABLE {table} ({", ".join(definitions)})')
                    conn.execute(f'INSERT INTO {table} SELECT DISTINCT * FROM {raw}')
                    conn.execute(f'DROP TABLE {raw}')
                    conn.execute('INSERT OR REPLACE INTO _group_stages VALUES (?,?)', (table, 'deduplicated'))
            location_id.cache_clear()
            log(f'{table}: building time-series index')
            with conn:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_loc_date ON {table}(unique_loc_id, date)')
                conn.execute('INSERT OR IGNORE INTO _completed_groups VALUES (?)', (table,))
                conn.execute('INSERT OR REPLACE INTO _group_stages VALUES (?,?)', (table, 'complete'))
            log(f'{table}: complete')
        log('Verifying database integrity and location references')
        integrity = [r[0] for r in conn.execute('PRAGMA integrity_check')]
        if integrity != ['ok']:
            raise ValueError(f'Integrity check failed: {integrity}')
        report = {'database': str(db), 'source': str(source), 'integrity_check': 'ok', 'tables': {}, 'files': []}
        for (table,) in conn.execute('SELECT table_group FROM _completed_groups ORDER BY table_group').fetchall():
            locs = table + '_unique_locs'
            stats = conn.execute(f'SELECT COUNT(*), MIN(date), MAX(date), MIN(lon), MAX(lon), MIN(lat), MAX(lat) FROM {table}').fetchone()
            invalid = conn.execute(f'SELECT COUNT(*) FROM {table} t LEFT JOIN {locs} l ON l.id=t.unique_loc_id WHERE l.id IS NULL OR l.lat != t.lat OR l.lon != t.lon').fetchone()[0]
            if invalid:
                raise ValueError(f'{table}: {invalid} invalid location references')
            incoming = conn.execute('SELECT SUM(rows_loaded) FROM _load_files WHERE table_group=?', (table,)).fetchone()[0]
            report['tables'][table] = dict(rows=stats[0], source_rows=incoming, duplicates_removed=incoming-stats[0], min_date=stats[1], max_date=stats[2], bounds=dict(min_lon=stats[3],max_lon=stats[4],min_lat=stats[5],max_lat=stats[6]), unique_locations=conn.execute(f'SELECT COUNT(*) FROM {locs}').fetchone()[0], invalid_location_references=invalid)
            log(f'{table}: {json.dumps(report["tables"][table])}')
        report['files'] = [dict(path=p, table=t, rows=n, complete=bool(c)) for p,t,n,c in conn.execute('SELECT * FROM _load_files ORDER BY path')]
        report_path = db.with_suffix('.validation.json')
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        log(f'Finished: {db}; validation: {report_path}')
        return report
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--batch-size', type=int, default=100_000)
    args = parser.parse_args()
    build(args.source, args.db, args.batch_size)
