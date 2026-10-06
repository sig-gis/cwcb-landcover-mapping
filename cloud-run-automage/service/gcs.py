"""Small GCS helpers shared by API service code."""
from __future__ import annotations

import json
from typing import Any

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Expected a gs:// URI")
    bucket, sep, name = uri[5:].partition("/")
    if not bucket or not sep or not name.strip("/"):
        raise ValueError("Expected gs://bucket/object")
    return bucket, name.strip("/")


def join_gs_uri(bucket: str, *parts: str) -> str:
    clean = [part.strip("/") for part in parts if part and part.strip("/")]
    return "gs://" + bucket + ("/" + "/".join(clean) if clean else "")


class GCS:
    def __init__(self, client: storage.Client | None = None):
        self.client = client or storage.Client()

    def exists(self, uri: str) -> bool:
        bucket, name = parse_gs_uri(uri)
        return self.client.bucket(bucket).blob(name).exists()

    def upload_json(self, uri: str, value: dict[str, Any], *, if_generation_match: int | None = None) -> None:
        bucket, name = parse_gs_uri(uri)
        options = {}
        if if_generation_match is not None:
            options["if_generation_match"] = if_generation_match
        self.client.bucket(bucket).blob(name).upload_from_string(
            json.dumps(value, indent=2, allow_nan=False),
            content_type="application/json",
            **options,
        )

    def download_json(self, uri: str) -> dict[str, Any]:
        bucket, name = parse_gs_uri(uri)
        return json.loads(self.client.bucket(bucket).blob(name).download_as_text())
