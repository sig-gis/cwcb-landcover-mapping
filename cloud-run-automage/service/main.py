"""FastAPI entrypoint for submitting and checking async AutoMage runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query

from .cloud_run_jobs import JobRunner
from .gcs import GCS, join_gs_uri
from .schemas import RunRequest, RunResponse, StatusResponse, validate_output_prefix
from .settings import Settings, load_settings

app = FastAPI(title="CWCB Landcover Mapping AutoMage API", version="0.1.0")


def get_settings() -> Settings:
    return load_settings()


def get_gcs() -> GCS:
    return GCS()


def get_job_runner() -> JobRunner:
    return JobRunner()


def run_uri_for(settings: Settings, output_prefix: str) -> str:
    return join_gs_uri(settings.output_bucket, settings.output_folder, output_prefix)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "cwcb-landcover-mapping", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/runs", response_model=RunResponse, status_code=202)
def create_run(
    request: RunRequest,
    settings: Settings = Depends(get_settings),
    gcs: GCS = Depends(get_gcs),
    jobs: JobRunner = Depends(get_job_runner),
) -> RunResponse:
    run_id = request.output_prefix
    run_uri = run_uri_for(settings, run_id)
    status_uri = run_uri + "/status.json"
    manifest_uri = run_uri + "/request.json"

    if gcs.exists(status_uri):
        raise HTTPException(status_code=409, detail=f"Output prefix already exists: {request.output_prefix}")

    dictionary_uri = request.dictionary_uri
    if request.dictionary is not None:
        dictionary_uri = run_uri + "/dictionary.json"
        gcs.upload_json(dictionary_uri, request.dictionary, if_generation_match=0)

    manifest = {
        "schema": "cwcb_automage_run_v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_tif_uri": settings.input_tif_uri,
        "run_uri": run_uri,
        "status_uri": status_uri,
        "dictionary_uri": dictionary_uri,
        "generate_superpixels": request.generate_superpixels,
        "config": {
            "wall_seconds": request.wall_seconds,
            "object_region_px": request.object_region_px,
            "object_compactness": request.object_compactness,
            "autocast_dtype": "float16",
        },
    }
    status = {
        "schema": "cwcb_automage_status_v1",
        "run_id": run_id,
        "state": "queued",
        "input_tif_uri": settings.input_tif_uri,
        "run_uri": run_uri,
        "manifest_uri": manifest_uri,
        "created_at": manifest["created_at"],
    }
    try:
        gcs.upload_json(manifest_uri, manifest, if_generation_match=0)
        gcs.upload_json(status_uri, status, if_generation_match=0)
        operation = jobs.run(settings.job_resource, manifest_uri)
        execution_name = getattr(getattr(operation, "metadata", None), "name", None) or getattr(operation, "operation", None)
        status["execution"] = str(execution_name) if execution_name else None
        gcs.upload_json(status_uri, status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue worker job: {type(exc).__name__}: {exc}") from exc

    return RunResponse(
        run_id=run_id,
        state="queued",
        input_tif_uri=settings.input_tif_uri,
        output_prefix=request.output_prefix,
        run_uri=run_uri,
        status_uri=status_uri,
        manifest_uri=manifest_uri,
    )


@app.get("/runs/status", response_model=StatusResponse)
def get_status(
    output_prefix: str = Query(...),
    settings: Settings = Depends(get_settings),
    gcs: GCS = Depends(get_gcs),
) -> StatusResponse:
    try:
        prefix = validate_output_prefix(output_prefix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    status_uri = run_uri_for(settings, prefix) + "/status.json"
    try:
        return StatusResponse(status=gcs.download_json(status_uri))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Status not found: {type(exc).__name__}: {exc}") from exc
