param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("check", "build-service", "build-worker", "deploy-service", "deploy-worker", "deploy-all", "deploy", "help")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $Root ".env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
        $name, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
    }
}

function Require-Env($Name) {
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) { throw "Set $Name in .env" }
    return $value
}

function Usage {
    Write-Host "Usage: .\deploy.ps1 -Command check|build-service|build-worker|deploy-service|deploy-worker|deploy-all"
}

if ($Command -eq "help") { Usage; exit 0 }

$Project = Require-Env "GCP_PROJECT"
$Region = Require-Env "GCP_REGION"
$InputTifUri = Require-Env "INPUT_TIF_URI"
$OutputBucket = Require-Env "OUTPUT_BUCKET"
$OutputFolder = Require-Env "OUTPUT_FOLDER"
$ApiServiceName = if ($env:API_SERVICE_NAME) { $env:API_SERVICE_NAME } else { "cwcb-landcover-api" }
$WorkerJobName = if ($env:WORKER_JOB_NAME) { $env:WORKER_JOB_NAME } else { "cwcb-landcover-worker" }
$ArtifactRepository = if ($env:ARTIFACT_REPOSITORY) { $env:ARTIFACT_REPOSITORY } else { "cwcb-landcover" }
$GpuType = if ($env:GPU_TYPE) { $env:GPU_TYPE } else { "nvidia-l4" }
$WorkerCpu = if ($env:WORKER_CPU) { $env:WORKER_CPU } else { "8" }
$WorkerMemory = if ($env:WORKER_MEMORY) { $env:WORKER_MEMORY } else { "32Gi" }
$WorkerTimeout = if ($env:WORKER_TIMEOUT) { $env:WORKER_TIMEOUT } else { "3600s" }
$RepoRoot = Split-Path -Parent $Root
$VendorAutomageRoot = Join-Path $RepoRoot "vendor\automage"

function Invoke-Gcloud($Args) {
    & gcloud --project=$Project --quiet @Args
    if ($LASTEXITCODE -ne 0) { throw "gcloud failed: $Args" }
}

function Test-Lfs {
    $script = "import json, sys`nfrom pathlib import Path`nmodel = Path(sys.argv[1]) / 'automage/models/sam3'`nindex = json.loads((model / 'model.safetensors.index.json').read_text())`nfor name in set(index['weight_map'].values()):`n    path = model / name`n    if not path.is_file() or path.stat().st_size < 1024:`n        raise SystemExit('Bundled weights missing. Run git lfs pull in automage before deploying.')`n"
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp -Value $script
    python $tmp $VendorAutomageRoot
    Remove-Item $tmp -Force
    if ($LASTEXITCODE -ne 0) { throw "LFS check failed" }
}

function New-ServiceBuildContext {
    $scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("cwcb-build-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    Copy-Item -Recurse -Path $Root -Destination (Join-Path $scratch "cloud-run-automage")
    Get-ChildItem -Path $scratch -Recurse -Force -Directory -Include __pycache__,.pytest_cache,.git | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    return $scratch
}

function New-WorkerBuildContext {
    $scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("cwcb-build-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    Copy-Item -Recurse -Path $Root -Destination (Join-Path $scratch "cloud-run-automage")
    New-Item -ItemType Directory -Path (Join-Path $scratch "vendor") | Out-Null
    Copy-Item -Recurse -Path $VendorAutomageRoot -Destination (Join-Path $scratch "vendor\automage")
    Get-ChildItem -Path $scratch -Recurse -Force -Directory -Include __pycache__,.pytest_cache,.git | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    return $scratch
}

function Ensure-Repo {
    Invoke-Gcloud @("services", "enable", "run.googleapis.com", "artifactregistry.googleapis.com", "cloudbuild.googleapis.com", "storage.googleapis.com", "iam.googleapis.com")
    & gcloud --project=$Project --quiet artifacts repositories describe $ArtifactRepository --location=$Region *> $null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Gcloud @("artifacts", "repositories", "create", $ArtifactRepository, "--location=$Region", "--repository-format=docker")
    }
}

function Build-Image($Kind, $Image) {
    if ($Kind -eq "worker") { $context = New-WorkerBuildContext } else { $context = New-ServiceBuildContext }
    try {
        Invoke-Gcloud @("builds", "submit", $context, "--region=$Region", "--config=$Root\cloudbuild.$Kind.yaml", "--substitutions=_IMAGE=$Image")
    } finally {
        Remove-Item -Recurse -Force $context -ErrorAction SilentlyContinue
    }
}

function Build-Service {
    Ensure-Repo
    $image = "$Region-docker.pkg.dev/$Project/$ArtifactRepository/api:$(Get-Date -AsUTC -Format yyyyMMdd-HHmmss)"
    Build-Image "service" $image
    Write-Output $image
}

function Build-Worker {
    Test-Lfs
    Ensure-Repo
    $image = "$Region-docker.pkg.dev/$Project/$ArtifactRepository/worker:$(Get-Date -AsUTC -Format yyyyMMdd-HHmmss)"
    Build-Image "worker" $image
    Write-Output $image
}

switch ($Command) {
    "check" {
        Invoke-Gcloud @("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
        Invoke-Gcloud @("projects", "describe", $Project, "--format=value(projectId,lifecycleState)")
        if (-not (Test-Path $VendorAutomageRoot)) { throw "Missing vendored AutoMage: $VendorAutomageRoot" }
        Test-Lfs
        Write-Host "Ready: project=$Project region=$Region input=$InputTifUri output=gs://$OutputBucket/$OutputFolder"
    }
    "build-service" { Build-Service }
    "build-worker" { Build-Worker }
    "deploy-service" {
        $serviceImage = Build-Service | Select-Object -Last 1
        $runtime = @()
        if ($env:RUNTIME_SERVICE_ACCOUNT) { $runtime += "--service-account=$env:RUNTIME_SERVICE_ACCOUNT" }
        Invoke-Gcloud (@("run", "deploy", $ApiServiceName, "--region=$Region", "--image=$serviceImage") + $runtime + @("--cpu=1", "--memory=1Gi", "--timeout=300s", "--allow-unauthenticated", "--set-env-vars=GCP_PROJECT=$Project,GCP_REGION=$Region,WORKER_JOB_NAME=$WorkerJobName,INPUT_TIF_URI=$InputTifUri,OUTPUT_BUCKET=$OutputBucket,OUTPUT_FOLDER=$OutputFolder"))
    }
    "deploy-worker" {
        $workerImage = Build-Worker | Select-Object -Last 1
        $runtime = @()
        if ($env:RUNTIME_SERVICE_ACCOUNT) { $runtime += "--service-account=$env:RUNTIME_SERVICE_ACCOUNT" }
        Invoke-Gcloud (@("run", "jobs", "deploy", $WorkerJobName, "--region=$Region", "--image=$workerImage") + $runtime + @("--gpu=1", "--gpu-type=$GpuType", "--no-gpu-zonal-redundancy", "--cpu=$WorkerCpu", "--memory=$WorkerMemory", "--tasks=1", "--parallelism=1", "--max-retries=0", "--task-timeout=$WorkerTimeout"))
    }
    "deploy" {
        $workerImage = Build-Worker | Select-Object -Last 1
        $serviceImage = Build-Service | Select-Object -Last 1
        $runtime = @()
        if ($env:RUNTIME_SERVICE_ACCOUNT) { $runtime += "--service-account=$env:RUNTIME_SERVICE_ACCOUNT" }
        Invoke-Gcloud (@("run", "jobs", "deploy", $WorkerJobName, "--region=$Region", "--image=$workerImage") + $runtime + @("--gpu=1", "--gpu-type=$GpuType", "--no-gpu-zonal-redundancy", "--cpu=$WorkerCpu", "--memory=$WorkerMemory", "--tasks=1", "--parallelism=1", "--max-retries=0", "--task-timeout=$WorkerTimeout"))
        Invoke-Gcloud (@("run", "deploy", $ApiServiceName, "--region=$Region", "--image=$serviceImage") + $runtime + @("--cpu=1", "--memory=1Gi", "--timeout=300s", "--allow-unauthenticated", "--set-env-vars=GCP_PROJECT=$Project,GCP_REGION=$Region,WORKER_JOB_NAME=$WorkerJobName,INPUT_TIF_URI=$InputTifUri,OUTPUT_BUCKET=$OutputBucket,OUTPUT_FOLDER=$OutputFolder"))
    }
    "deploy-all" {
        $workerImage = Build-Worker | Select-Object -Last 1
        $serviceImage = Build-Service | Select-Object -Last 1
        $runtime = @()
        if ($env:RUNTIME_SERVICE_ACCOUNT) { $runtime += "--service-account=$env:RUNTIME_SERVICE_ACCOUNT" }
        Invoke-Gcloud (@("run", "jobs", "deploy", $WorkerJobName, "--region=$Region", "--image=$workerImage") + $runtime + @("--gpu=1", "--gpu-type=$GpuType", "--no-gpu-zonal-redundancy", "--cpu=$WorkerCpu", "--memory=$WorkerMemory", "--tasks=1", "--parallelism=1", "--max-retries=0", "--task-timeout=$WorkerTimeout"))
        Invoke-Gcloud (@("run", "deploy", $ApiServiceName, "--region=$Region", "--image=$serviceImage") + $runtime + @("--cpu=1", "--memory=1Gi", "--timeout=300s", "--allow-unauthenticated", "--set-env-vars=GCP_PROJECT=$Project,GCP_REGION=$Region,WORKER_JOB_NAME=$WorkerJobName,INPUT_TIF_URI=$InputTifUri,OUTPUT_BUCKET=$OutputBucket,OUTPUT_FOLDER=$OutputFolder"))
    }
}