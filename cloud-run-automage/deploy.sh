#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$root/.." && pwd)
vendor_automage_root="$repo_root/vendor/automage"

if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a
fi

: "${GCP_PROJECT:?Set GCP_PROJECT in .env}"
: "${GCP_REGION:?Set GCP_REGION in .env}"
: "${INPUT_TIF_URI:?Set INPUT_TIF_URI in .env}"
: "${OUTPUT_BUCKET:?Set OUTPUT_BUCKET in .env}"
: "${OUTPUT_FOLDER:?Set OUTPUT_FOLDER in .env}"
: "${API_SERVICE_NAME:=cwcb-landcover-api}"
: "${WORKER_JOB_NAME:=cwcb-landcover-worker}"
: "${ARTIFACT_REPOSITORY:=cwcb-landcover}"
: "${GPU_TYPE:=nvidia-l4}"
: "${WORKER_CPU:=8}"
: "${WORKER_MEMORY:=32Gi}"
: "${WORKER_TIMEOUT:=3600s}"

gc=(gcloud --project="$GCP_PROJECT" --quiet)

usage() {
  cat <<EOF
Usage: ./deploy.sh COMMAND

Commands:
  check          Validate local config and cloud access
  build-service   Build and push API service image only
  build-worker    Build and push GPU worker image only
  deploy-service  Build and deploy API service only
  deploy-worker   Build and deploy GPU worker job only
  deploy-all      Enable services, build images, deploy worker job and API service
EOF
}

scratch=''
cleanup() { [[ -n "$scratch" ]] && rm -rf "$scratch"; }
trap cleanup EXIT

copy_service_context() {
  scratch=$(mktemp -d)
  mkdir -p "$scratch/cloud-run-automage"
  cp -R "$root"/* "$scratch/cloud-run-automage/"
  find "$scratch" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .git \) -prune -exec rm -rf {} + 2>/dev/null || true
  echo "$scratch"
}

copy_worker_context() {
  scratch=$(mktemp -d)
  mkdir -p "$scratch/cloud-run-automage" "$scratch/vendor"
  cp -R "$root"/* "$scratch/cloud-run-automage/"
  cp -R "$vendor_automage_root" "$scratch/vendor/automage"
  find "$scratch" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .git \) -prune -exec rm -rf {} + 2>/dev/null || true
  echo "$scratch"
}

check_lfs() {
  python3 - "$vendor_automage_root" <<'PY'
import json, sys
from pathlib import Path
model = Path(sys.argv[1]) / 'automage/models/sam3'
index = json.loads((model / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    path = model / name
    if not path.is_file() or path.stat().st_size < 1024:
        raise SystemExit('Bundled weights missing. Run git lfs pull in automage before deploying.')
PY
}

check() {
  "${gc[@]}" auth list --filter=status:ACTIVE --format='value(account)'
  "${gc[@]}" projects describe "$GCP_PROJECT" --format='value(projectId,lifecycleState)'
  [[ -d "$vendor_automage_root" ]] || { echo "Missing vendored AutoMage: $vendor_automage_root" >&2; return 1; }
  check_lfs
  printf 'Ready: project=%s region=%s input=%s output=gs://%s/%s\n' "$GCP_PROJECT" "$GCP_REGION" "$INPUT_TIF_URI" "$OUTPUT_BUCKET" "$OUTPUT_FOLDER"
}

ensure_repo() {
  "${gc[@]}" services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com iam.googleapis.com
  if ! "${gc[@]}" artifacts repositories describe "$ARTIFACT_REPOSITORY" --location="$GCP_REGION" >/dev/null 2>&1; then
    "${gc[@]}" artifacts repositories create "$ARTIFACT_REPOSITORY" --location="$GCP_REGION" --repository-format=docker
  fi
}

build_image() {
  local kind=$1 image=$2 context
  if [[ "$kind" == worker ]]; then
    context=$(copy_worker_context)
  else
    context=$(copy_service_context)
  fi
  "${gc[@]}" builds submit "$context" --region="$GCP_REGION" \
    --config="$root/cloudbuild.${kind}.yaml" --substitutions="_IMAGE=$image"
}

build_service() {
  ensure_repo
  image="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPOSITORY}/api:$(date -u +%Y%m%d-%H%M%S)"
  build_image service "$image"
  echo "$image"
}

build_worker() {
  check_lfs
  ensure_repo
  image="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REPOSITORY}/worker:$(date -u +%Y%m%d-%H%M%S)"
  build_image worker "$image"
  echo "$image"
}

deploy_all() {
  check
  deploy_worker
  deploy_service
}

deploy_service() {
  service_image=$(build_service | tail -n1)
  runtime_flags=()
  if [[ -n "${RUNTIME_SERVICE_ACCOUNT:-}" ]]; then runtime_flags+=(--service-account="$RUNTIME_SERVICE_ACCOUNT"); fi
  "${gc[@]}" run deploy "$API_SERVICE_NAME" --region="$GCP_REGION" --image="$service_image" \
    "${runtime_flags[@]}" --cpu=1 --memory=1Gi --timeout=300s --allow-unauthenticated \
    --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION,WORKER_JOB_NAME=$WORKER_JOB_NAME,INPUT_TIF_URI=$INPUT_TIF_URI,OUTPUT_BUCKET=$OUTPUT_BUCKET,OUTPUT_FOLDER=$OUTPUT_FOLDER"
}

deploy_worker() {
  worker_image=$(build_worker | tail -n1)
  runtime_flags=()
  if [[ -n "${RUNTIME_SERVICE_ACCOUNT:-}" ]]; then runtime_flags+=(--service-account="$RUNTIME_SERVICE_ACCOUNT"); fi
  "${gc[@]}" run jobs deploy "$WORKER_JOB_NAME" --region="$GCP_REGION" --image="$worker_image" \
    "${runtime_flags[@]}" --gpu=1 --gpu-type="$GPU_TYPE" --no-gpu-zonal-redundancy \
    --cpu="$WORKER_CPU" --memory="$WORKER_MEMORY" --tasks=1 --parallelism=1 --max-retries=0 --task-timeout="$WORKER_TIMEOUT"
}

case "${1:-help}" in
  check) check ;;
  build-service) build_service ;;
  build-worker) build_worker ;;
  deploy-service) deploy_service ;;
  deploy-worker) deploy_worker ;;
  deploy-all|deploy) deploy_all ;;
  help|--help|-h) usage ;;
  *) usage >&2; exit 1 ;;
esac
