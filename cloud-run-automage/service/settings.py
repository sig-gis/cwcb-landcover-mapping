"""Environment-backed settings for the API service."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project: str
    region: str
    worker_job_name: str
    input_tif_uri: str
    output_bucket: str
    output_folder: str

    @property
    def job_resource(self) -> str:
        return f"projects/{self.project}/locations/{self.region}/jobs/{self.worker_job_name}"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        project=_required("GCP_PROJECT"),
        region=_required("GCP_REGION"),
        worker_job_name=_required("WORKER_JOB_NAME"),
        input_tif_uri=_required("INPUT_TIF_URI"),
        output_bucket=_required("OUTPUT_BUCKET"),
        output_folder=os.environ.get("OUTPUT_FOLDER", "").strip().strip("/"),
    )
