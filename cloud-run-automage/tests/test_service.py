import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from service.main import app, get_gcs, get_job_runner, get_settings
from service.settings import Settings


class MemoryGCS:
    def __init__(self):
        self.objects = {}

    def exists(self, uri):
        return uri in self.objects

    def upload_json(self, uri, value, if_generation_match=None):
        if if_generation_match == 0 and uri in self.objects:
            raise RuntimeError("object exists")
        self.objects[uri] = json.loads(json.dumps(value))

    def download_json(self, uri):
        return self.objects[uri]


class Jobs:
    def __init__(self):
        self.calls = []

    def run(self, job_name, manifest_uri):
        self.calls.append((job_name, manifest_uri))

        class Operation:
            operation = "operations/example"

        return Operation()


@pytest.fixture()
def client():
    gcs = MemoryGCS()
    jobs = Jobs()
    settings = Settings(
        project="test-project",
        region="us-east4",
        worker_job_name="worker-job",
        input_tif_uri="gs://inputs/demo.tif",
        output_bucket="outputs",
        output_folder="automage",
    )
    app.dependency_overrides[get_gcs] = lambda: gcs
    app.dependency_overrides[get_job_runner] = lambda: jobs
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield TestClient(app), gcs, jobs
    finally:
        app.dependency_overrides.clear()


def test_create_run_uses_env_input_and_bucket_with_prefix(client):
    http, gcs, jobs = client
    response = http.post("/runs", json={"output_prefix": "demo/run_001", "generate_superpixels": True})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["run_uri"] == "gs://outputs/automage/demo/run_001"
    assert body["input_tif_uri"] == "gs://inputs/demo.tif"
    manifest = gcs.objects["gs://outputs/automage/demo/run_001/request.json"]
    assert manifest["input_tif_uri"] == "gs://inputs/demo.tif"
    assert manifest["generate_superpixels"] is True
    assert manifest["dictionary_uri"] is None
    assert jobs.calls == [("projects/test-project/locations/us-east4/jobs/worker-job", body["manifest_uri"])]


def test_inline_dictionary_is_uploaded_and_referenced(client):
    http, gcs, _ = client
    dictionary = {"classes": []}
    response = http.post("/runs", json={"output_prefix": "run2", "dictionary": dictionary})
    assert response.status_code == 202, response.text
    assert gcs.objects["gs://outputs/automage/run2/dictionary.json"] == dictionary
    assert gcs.objects["gs://outputs/automage/run2/request.json"]["dictionary_uri"] == "gs://outputs/automage/run2/dictionary.json"


def test_rejects_two_dictionary_sources(client):
    http, _, _ = client
    response = http.post("/runs", json={"output_prefix": "run3", "dictionary": {"classes": []}, "dictionary_uri": "gs://b/d.json"})
    assert response.status_code == 422


@pytest.mark.parametrize("prefix", ["gs://bucket/path", "../bad", "/bad", "bad/", "bad//path", "bad path"])
def test_rejects_bad_prefixes(client, prefix):
    http, _, _ = client
    response = http.post("/runs", json={"output_prefix": prefix})
    assert response.status_code == 422


def test_status_uses_env_bucket_and_folder(client):
    http, gcs, _ = client
    gcs.objects["gs://outputs/automage/demo/status.json"] = {"state": "complete"}
    response = http.get("/runs/status", params={"output_prefix": "demo"})
    assert response.status_code == 200
    assert response.json()["status"]["state"] == "complete"
