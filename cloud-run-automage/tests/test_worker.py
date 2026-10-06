import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.main import run_manifest


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.dictionary = "bundled-default"


class MemoryGCS:
    def __init__(self):
        self.jsons = {}
        self.files = {}
        self.uploaded = {}

    def download_json(self, uri):
        return self.jsons[uri]

    def upload_json(self, uri, value):
        self.jsons[uri] = json.loads(json.dumps(value))

    def download_file(self, uri, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[uri])

    def upload_directory(self, source, destination_uri):
        self.uploaded[destination_uri] = sorted(str(p.relative_to(source)).replace("\\", "/") for p in source.rglob("*") if p.is_file())


def manifest(dictionary_uri=None, generate_superpixels=True):
    return {
        "run_id": "demo",
        "input_tif_uri": "gs://inputs/demo.tif",
        "run_uri": "gs://outputs/automage/demo",
        "status_uri": "gs://outputs/automage/demo/status.json",
        "dictionary_uri": dictionary_uri,
        "generate_superpixels": generate_superpixels,
        "config": {"wall_seconds": 3000, "object_region_px": 28, "object_compactness": 12.0, "autocast_dtype": "float16"},
    }


def test_worker_uses_custom_dictionary_and_superpixel_flag():
    gcs = MemoryGCS()
    gcs.jsons["gs://outputs/automage/demo/request.json"] = manifest(dictionary_uri="gs://dict/classes.json", generate_superpixels=True)
    gcs.files["gs://inputs/demo.tif"] = b"tif"
    gcs.files["gs://dict/classes.json"] = b'{"classes": []}'
    seen = {}

    def classify(source, out, config):
        seen["source"] = source.read_bytes()
        seen["objects"] = config.objects
        seen["dictionary"] = Path(config.dictionary).read_bytes()
        out.mkdir()
        (out / "summary.json").write_text("{}")
        work = out.with_name(out.name + ".work")
        work.mkdir()
        (work / "provenance.json").write_text("{}")
        return {"state": "complete", "feature_count": 4, "seconds": 2.5}

    summary = run_manifest("gs://outputs/automage/demo/request.json", gcs=gcs, classify_func=classify, config_factory=Config)
    assert summary["state"] == "complete"
    assert seen == {"source": b"tif", "objects": True, "dictionary": b'{"classes": []}'}
    assert "summary.json" in gcs.uploaded["gs://outputs/automage/demo/result"]
    assert "provenance.json" in gcs.uploaded["gs://outputs/automage/demo/result.work"]
    assert gcs.jsons["gs://outputs/automage/demo/status.json"]["state"] == "complete"


def test_worker_uses_bundled_dictionary_when_none_supplied():
    gcs = MemoryGCS()
    gcs.jsons["gs://outputs/automage/demo/request.json"] = manifest(dictionary_uri=None, generate_superpixels=False)
    gcs.files["gs://inputs/demo.tif"] = b"tif"
    seen = {}

    def classify(source, out, config):
        seen["objects"] = config.objects
        seen["dictionary"] = config.dictionary
        out.mkdir()
        return {"state": "complete"}

    run_manifest("gs://outputs/automage/demo/request.json", gcs=gcs, classify_func=classify, config_factory=Config)
    assert seen == {"objects": False, "dictionary": "bundled-default"}


def test_worker_reports_failure():
    gcs = MemoryGCS()
    gcs.jsons["gs://outputs/automage/demo/request.json"] = manifest()
    gcs.files["gs://inputs/demo.tif"] = b"tif"

    def classify(*args):
        raise RuntimeError("GPU failed")

    with pytest.raises(RuntimeError, match="GPU failed"):
        run_manifest("gs://outputs/automage/demo/request.json", gcs=gcs, classify_func=classify, config_factory=Config)
    assert gcs.jsons["gs://outputs/automage/demo/status.json"]["state"] == "failed"
    assert "GPU failed" in gcs.jsons["gs://outputs/automage/demo/status.json"]["error"]
