# CWCB Landcover AutoMage Cloud Run

Async Cloud Run deployment for AutoMage land-cover mapping.

## Architecture

- **API service**: FastAPI HTTP service. It accepts a run request, writes a manifest/status JSON to GCS, triggers a GPU Cloud Run Job, and returns immediately.
- **Worker job**: GPU Cloud Run Job. It downloads the configured input TIFF from GCS, runs AutoMage, and uploads outputs to GCS.

## Configuration

Copy `env.example` to `.env` and edit locally. Do not commit `.env`.

Required values:

```bash
GCP_PROJECT=your-project-id
GCP_REGION=us-east4
INPUT_TIF_URI=gs://your-input-bucket/path/demo.tif
OUTPUT_BUCKET=your-output-bucket
OUTPUT_FOLDER=output_folder
SAM3_WEIGHTS_GCS_URI=gs://your-model-bucket/automage/sam3
```

The API always uses `INPUT_TIF_URI` for now. Users provide only `output_prefix` in API requests. Outputs go to:

```text
gs://$OUTPUT_BUCKET/$OUTPUT_FOLDER/$output_prefix/
```

## API

### Health

`GET /health`

### Start a run

`POST /runs`

```json
{
  "output_prefix": "demo/boulder_001",
  "generate_superpixels": true
}
```

Optional custom dictionary from GCS:

```json
{
  "output_prefix": "demo/boulder_002",
  "dictionary_uri": "gs://your-bucket/dictionaries/classes.json",
  "generate_superpixels": false
}
```

Optional inline dictionary:

```json
{
  "output_prefix": "demo/boulder_003",
  "dictionary": {"classes": []},
  "generate_superpixels": true
}
```

If no dictionary is supplied, the bundled AutoMage dictionary is used.

### Status

Use a query parameter because prefixes may contain slashes:

`GET /runs/status?output_prefix=demo/boulder_001`

## Output layout

```text
gs://$OUTPUT_BUCKET/$OUTPUT_FOLDER/$output_prefix/
  request.json
  status.json
  dictionary.json          # only for inline custom dictionary
  result/
    summary.json
    classification.gpkg
    overview.png
    features/
    objects.tif            # when generate_superpixels=true
  result.work/
    provenance.json
    input.json
    classes.tif
    concepts.tif
    scores.tif
```

## Deploy

Linux/macOS/Git Bash:

```bash
./deploy.sh check
./deploy.sh deploy-worker
./deploy.sh deploy-service
```

PowerShell:

```powershell
.\deploy.ps1 -Command check
.\deploy.ps1 -Command deploy-worker
.\deploy.ps1 -Command deploy-service
```

The worker image installs the vendored AutoMage package from `vendor/automage` inside this repository. API-only deploys use only `cloud-run-automage/`; worker deploys use `cloud-run-automage/` plus `vendor/automage/` and download SAM3 weights from `SAM3_WEIGHTS_GCS_URI` during Cloud Build.

## IAM notes

This deployment uses two configurable service accounts:

- `BUILD_SERVICE_ACCOUNT`: used by `gcloud builds submit` when building/pushing API and worker images.
- `RUNTIME_SERVICE_ACCOUNT`: attached to both the Cloud Run API service and Cloud Run worker job.