param(
    [string]$PythonExe = "python",
    [int]$Epochs = 250,
    [ValidateSet("all", "directfiles", "binary", "polygons")]
    [string]$Strategy3D = "binary",
    [int]$BatchSize3D = 1,
    [int]$NumWorkers = 0,
    [string]$SamplingMode3D = "class-aware",
    [double]$ForegroundBoost3D = 2.0,
    [string]$ClassBoosts3D = "",
    [double]$MaxSampleWeight3D = 12,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$root = Get-Location
$logsDir = Join-Path $root "results/run_logs/unet3d_cpu_5folds"
$markersDir = Join-Path $root "results/run_markers/unet3d_cpu_5folds"
$plotsDir = Join-Path $root "results/plots/unet3d_250_cpu_5folds"

New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $markersDir -Force | Out-Null
New-Item -ItemType Directory -Path $plotsDir -Force | Out-Null

Write-Host "Python: $PythonExe"
Write-Host "Epochs: $Epochs"
Write-Host "Strategy3D: $Strategy3D"
Write-Host "BatchSize3D: $BatchSize3D"
Write-Host "NumWorkers: $NumWorkers"
Write-Host "SamplingMode3D: $SamplingMode3D"
Write-Host "Resume mode: $Resume"

# Force CPU mode even if a CUDA GPU exists on the machine.
$env:CUDA_VISIBLE_DEVICES = ""

for ($i = 1; $i -le 5; $i++) {
    $fold = "fold_$i"
    $marker = Join-Path $markersDir "train_${Strategy3D}_$fold.done"
    $logFile = Join-Path $logsDir "train_${Strategy3D}_$fold.log"
    $outputJson = "results/unet3d_train_results_${Strategy3D}_$fold.json"

    if ($Resume -and (Test-Path $marker)) {
        Write-Host "[resume] Skip $fold (marker exists)"
        continue
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "TRAIN UNet3D | strategy=$Strategy3D | $fold | epochs=$Epochs | CPU"
    Write-Host "============================================================"

    $cmd = @(
        $PythonExe,
        "4_Unet3D/train_unet3d_compare.py",
        "--strategy", $Strategy3D,
        "--fold", $fold,
        "--epochs", "$Epochs",
        "--num-classes", "41",
        "--batch-size", "$BatchSize3D",
        "--num-workers", "$NumWorkers",
        "--sampling-mode", $SamplingMode3D,
        "--foreground-boost", "$ForegroundBoost3D",
        "--max-sample-weight", "$MaxSampleWeight3D",
        "--output-json", $outputJson
    )

    if (-not [string]::IsNullOrWhiteSpace($ClassBoosts3D)) {
        $cmd += @("--class-boosts", $ClassBoosts3D)
    }

    $start = Get-Date
    & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($exitCode -ne 0) {
        throw "Training failed on $fold (exit=$exitCode). See $logFile"
    }

    New-Item -ItemType File -Path $marker -Force | Out-Null
    Write-Host "[ok] $fold completed in $([math]::Round($elapsed.TotalMinutes, 2)) min"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Generate curves from all folds"
Write-Host "============================================================"

$plotLog = Join-Path $logsDir "plot_curves.log"
$plotCmd = @(
    $PythonExe,
    "4_Unet3D/plot_unet3d_curves.py",
    "--input-glob", "results/unet3d_train_results_${Strategy3D}_fold_*.json",
    "--output-dir", "results/plots/unet3d_250_cpu_5folds_${Strategy3D}"
)

& $plotCmd[0] $plotCmd[1..($plotCmd.Length - 1)] 2>&1 | Tee-Object -FilePath $plotLog -Append
$plotExit = $LASTEXITCODE
if ($plotExit -ne 0) {
    throw "Curve generation failed (exit=$plotExit). See $plotLog"
}

Write-Host ""
Write-Host "Done. Outputs:"
Write-Host "- Fold JSONs: results/unet3d_train_results_${Strategy3D}_fold_*.json"
Write-Host "- Logs: results/run_logs/unet3d_cpu_5folds"
Write-Host "- Markers: results/run_markers/unet3d_cpu_5folds"
Write-Host "- Curves: results/plots/unet3d_250_cpu_5folds_${Strategy3D}"
