"""Request/response models and safe output-prefix validation."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_output_prefix(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("output_prefix is required")
    if value.startswith("gs://"):
        raise ValueError("output_prefix must be a relative GCS prefix, not a gs:// URI")
    if value.startswith("/") or value.endswith("/"):
        raise ValueError("output_prefix must not start or end with slash")
    if "\\" in value:
        raise ValueError("output_prefix must use forward slashes only")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("output_prefix contains an invalid path segment")
    if not all(_SEGMENT.fullmatch(part) for part in parts):
        raise ValueError("output_prefix may contain only letters, numbers, slash, dash, underscore, and period")
    return value


class RunRequest(BaseModel):
    output_prefix: str = Field(..., description="Relative output prefix under OUTPUT_BUCKET/OUTPUT_FOLDER")
    generate_superpixels: bool = False
    dictionary_uri: str | None = None
    dictionary: dict[str, Any] | None = None
    wall_seconds: int = Field(3000, ge=600, le=21600)
    object_region_px: int = Field(28, ge=2, le=1024)
    object_compactness: float = Field(12.0, gt=0, le=1000)

    @field_validator("output_prefix")
    @classmethod
    def _validate_output_prefix(cls, value: str) -> str:
        return validate_output_prefix(value)

    @field_validator("dictionary_uri")
    @classmethod
    def _validate_dictionary_uri(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("gs://"):
            raise ValueError("dictionary_uri must be a gs:// URI")
        return value

    @model_validator(mode="after")
    def _one_dictionary_source(self) -> "RunRequest":
        if self.dictionary is not None and self.dictionary_uri is not None:
            raise ValueError("Supply either dictionary or dictionary_uri, not both")
        return self


class RunResponse(BaseModel):
    run_id: str
    state: str
    input_tif_uri: str
    output_prefix: str
    run_uri: str
    status_uri: str
    manifest_uri: str


class StatusResponse(BaseModel):
    status: dict[str, Any]
