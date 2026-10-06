"""GCS helpers for the GPU worker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Expected a gs:// URI")
    bucket, sep, name = uri[5:].partition("/")
    if not bucket or not sep or not name.strip("/"):
        raise ValueError("Expected gs://bucket/object")
    return bucket, name.strip("/")


class GCS:
    def __init__(self, client: storage.Client | None = None):
        self.client = client or storage.Client()

    def download_file(self, uri: str, destination: Path) -> None:
        bucket, name = parse_gs_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.bucket(bucket).blob(name).download_to_filename(destination)

    def download_json(self, uri: str) -> dict[str, Any]:
        bucket, name = parse_gs_uri(uri)
        return json.loads(self.client.bucket(bucket).blob(name).download_as_text())

    def upload_json(self, uri: str, value: dict[str, Any]) -> None:
        bucket, name = parse_gs_uri(uri)
        self.client.bucket(bucket).blob(name).upload_from_string(
            json.dumps(value, indent=2, allow_nan=False),
            content_type="application/json",
        )

    def upload_directory(self, source: Path, destination_uri: str) -> None:
        bucket_name, prefix = parse_gs_uri(destination_uri.rstrip("/") + "/placeholder")
        prefix = prefix.rsplit("/", 1)[0]
        bucket = self.client.bucket(bucket_name)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                name = prefix + "/" + path.relative_to(source).as_posix()
                bucket.blob(name).upload_from_filename(path)
