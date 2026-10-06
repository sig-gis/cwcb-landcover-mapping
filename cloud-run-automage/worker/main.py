"""GPU worker that executes one AutoMage run from a GCS request manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Callable

from .gcs import GCS
from .settings import load_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(manifest: dict[str, Any], state: str, **details: Any) -> dict[str, Any]:
    return {
        "schema": "cwcb_automage_status_v1",
        "run_id": manifest["run_id"],
        "state": state,
        "input_tif_uri": manifest["input_tif_uri"],
        "run_uri": manifest["run_uri"],
        "updated_at": _now(),
        **details,
    }


def run_manifest(
    manifest_uri: str,
    *,
    gcs: GCS | None = None,
    classify_func: Callable | None = None,
    config_factory: Callable | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    gcs = gcs or GCS()
    manifest = gcs.download_json(manifest_uri)
    status_uri = manifest.get("status_uri") or manifest["run_uri"].rstrip("/") + "/status.json"
    gcs.upload_json(status_uri, _status(manifest, "running", manifest_uri=manifest_uri))

    with tempfile.TemporaryDirectory(prefix="cwcb-automage-") as directory:
        root = Path(directory)
        input_path = root / "input.tif"
        result_dir = root / "result"
        try:
            gcs.download_file(manifest["input_tif_uri"], input_path)
            dictionary_uri = manifest.get("dictionary_uri")
            dictionary_path = None
            if dictionary_uri:
                dictionary_path = root / "dictionary.json"
                gcs.download_file(dictionary_uri, dictionary_path)

            if classify_func is None or config_factory is None:
                from automage import Config, classify
                classify_func = classify_func or classify
                config_factory = config_factory or Config

            cfg_values = manifest.get("config", {})
            config = config_factory(
                objects=bool(manifest.get("generate_superpixels", False)),
                autocast_dtype=cfg_values.get("autocast_dtype", "float16"),
                wall_seconds=int(cfg_values.get("wall_seconds", settings.default_wall_seconds)),
                object_region_px=int(cfg_values.get("object_region_px", 28)),
                object_compactness=float(cfg_values.get("object_compactness", 12.0)),
            )
            if dictionary_path is not None:
                config.dictionary = str(dictionary_path)

            summary = classify_func(input_path, result_dir, config)
            work_dir = result_dir.with_name(result_dir.name + ".work")
            if result_dir.exists():
                gcs.upload_directory(result_dir, manifest["run_uri"].rstrip("/") + "/result")
            if work_dir.exists():
                gcs.upload_directory(work_dir, manifest["run_uri"].rstrip("/") + "/result.work")

            state = "complete" if summary.get("state") == "complete" else "incomplete"
            final = _status(
                manifest,
                state,
                manifest_uri=manifest_uri,
                feature_count=summary.get("feature_count"),
                seconds=summary.get("seconds"),
                result_uri=manifest["run_uri"].rstrip("/") + "/result",
                work_uri=manifest["run_uri"].rstrip("/") + "/result.work",
                summary=summary,
            )
            gcs.upload_json(status_uri, final)
            if state != "complete":
                raise RuntimeError("Analysis did not complete: " + str(summary.get("state")))
            return summary
        except Exception as exc:
            gcs.upload_json(status_uri, _status(manifest, "failed", manifest_uri=manifest_uri, error=f"{type(exc).__name__}: {exc}"))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="GCS URI to request.json")
    args = parser.parse_args()
    run_manifest(args.manifest)


if __name__ == "__main__":
    main()
