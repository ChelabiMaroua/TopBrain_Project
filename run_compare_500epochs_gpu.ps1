param(
    [string]$PythonExe = "python",
    [string]$Fold = "fold_1",
    [int]$BatchSize2D = 8,
    [int]$BatchSize3D = 1,
    [int]$NumWorkers = 4,
    [string]$GpuId = "0",
    [switch]$RunEtl2D,
    [switch]$Skip2D,
    [switch]$Skip3D,
    [switch]$SkipCompare,
    [switch]$Resume,
    [string]$SamplingMode2D = "class-aware",
    [string]$SamplingMode3D = "class-aware",
    [double]$ForegroundBoost2D = 2.5,
    [double]$ForegroundBoost3D = 2.0,
    [string]$ClassBoosts2D = "",
    [string]$ClassBoosts3D = "",
    [double]$MaxSampleWeight2D = 14,
    [double]$MaxSampleWeight3D = 12
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command,
        [string]$Marker,
        [string]$LogPath,
        [switch]$Skip
    )

    if ($Skip) {
        Write-Host "[skip] $Name"
        return
    }

    if ($Resume -and (Test-Path $Marker)) {
        Write-Host "[resume] marker found, skipping $Name"
        return
    }

    Write-Host "\n============================================================"
    Write-Host "STEP: $Name"
    Write-Host "CMD : $($Command -join ' ')"
    Write-Host "LOG : $LogPath"
    Write-Host "============================================================"

    $start = Get-Date
    & $Command[0] $Command[1..($Command.Length - 1)] 2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($exitCode -ne 0) {
        throw "Step failed: $Name (exit=$exitCode). Check log: $LogPath"
    }

    New-Item -ItemType File -Path $Marker -Force | Out-Null
    Write-Host "[ok] $Name finished in $([math]::Round($elapsed.TotalMinutes,2)) min"
}

Write-Host "Using Python: $PythonExe"
Write-Host "Fold: $Fold"
Write-Host "2D epochs: 500 | 3D epochs: 500"
Write-Host "Batch sizes -> 2D: $BatchSize2D | 3D: $BatchSize3D"
Write-Host "Num workers: $NumWorkers"
Write-Host "Sampling 2D: $SamplingMode2D | class boosts: $ClassBoosts2D"
Write-Host "Sampling 3D: $SamplingMode3D | class boosts: $ClassBoosts3D"
Write-Host "Resume mode: $Resume"

# Pin run to one GPU (edit/remove if multi-GPU scheduling differs)
$env:CUDA_VISIBLE_DEVICES = $GpuId

$root = Get-Location
$logsDir = Join-Path $root "results\run_logs"
$markersDir = Join-Path $root "results\run_markers"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $markersDir -Force | Out-Null

# Optional ETL 2D step (safe to rerun; loader is idempotent)
$etlCmd = @(
    $PythonExe,
    "1_ETL/Load/load_t6_mongodb_insert_2d.py",
    "--target-size", "256", "256", "192",
    "--class-min", "0",
    "--class-max", "40",
    "--num-classes", "41",
    "--window-min", "0",
    "--window-max", "600"
)
Invoke-Step -Name "Populate 2D Mongo collections" -Command $etlCmd -Marker (Join-Path $markersDir "step_etl2d.done") -LogPath (Join-Path $logsDir "step_etl2d.log") -Skip:(!$RunEtl2D)

$train2dCmd = @(
    $PythonExe,
    "4_Unet2D/train_unet2d_compare.py",
    "--strategy", "all",
    "--fold", $Fold,
    "--epochs", "500",
    "--num-classes", "41",
    "--batch-size", "$BatchSize2D",
    "--num-workers", "$NumWorkers",
    "--augment",
    "--sampling-mode", $SamplingMode2D,
    "--foreground-boost", "$ForegroundBoost2D",
    "--class-boosts", $ClassBoosts2D,
    "--max-sample-weight", "$MaxSampleWeight2D"
)
Invoke-Step -Name "Train UNet2D (all strategies, 500 epochs)" -Command $train2dCmd -Marker (Join-Path $markersDir "step_train2d.done") -LogPath (Join-Path $logsDir "step_train2d.log") -Skip:$Skip2D

$train3dCmd = @(
    $PythonExe,
    "4_Unet3D/train_unet3d_compare.py",
    "--strategy", "all",
    "--fold", $Fold,
    "--epochs", "500",
    "--num-classes", "41",
    "--batch-size", "$BatchSize3D",
    "--num-workers", "$NumWorkers",
    "--sampling-mode", $SamplingMode3D,
    "--foreground-boost", "$ForegroundBoost3D",
    "--class-boosts", $ClassBoosts3D,
    "--max-sample-weight", "$MaxSampleWeight3D"
)
Invoke-Step -Name "Train UNet3D (all strategies, 500 epochs)" -Command $train3dCmd -Marker (Join-Path $markersDir "step_train3d.done") -LogPath (Join-Path $logsDir "step_train3d.log") -Skip:$Skip3D

$compareCmd = @(
    $PythonExe,
    "compare_unet2d_unet3d.py",
    "--unet2d-json", "results/unet2d_train_results.json",
    "--unet3d-json", "results/unet3d_train_results.json",
    "--output-json", "results/unet_compare_summary.json",
    "--output-md", "results/unet_compare_summary.md"
)
Invoke-Step -Name "Build 2D vs 3D comparison report" -Command $compareCmd -Marker (Join-Path $markersDir "step_compare.done") -LogPath (Join-Path $logsDir "step_compare.log") -Skip:$SkipCompare

Write-Host "Done."
Write-Host "2D results: results/unet2d_train_results.json"
Write-Host "3D results: results/unet3d_train_results.json"
Write-Host "Comparison: results/unet_compare_summary.md"
Write-Host "Logs: results/run_logs"
Write-Host "Resume markers: results/run_markers"
