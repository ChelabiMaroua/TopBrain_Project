param(
    [int]$DatasetId = 501,
    [string]$Configuration = "3d_fullres",
    [int]$Fold = 0,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda",
    [int]$NProcDA = 2,
    [switch]$Npz,
    [switch]$Continue,
    [switch]$DryRun,
    [string]$NnUNetRaw = "nnUNet_raw",
    [string]$NnUNetPreprocessed = "nnUNet_preprocessed",
    [string]$NnUNetResults = "nnUNet_results"
)

$ErrorActionPreference = "Stop"

# Constrain thread pools to reduce RAM pressure and prevent sporadic DLL/memory failures.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# nnUNet data-augmentation workers. Lower values reduce RAM usage (slower but safer).
$env:nnUNet_n_proc_DA = [string]$NProcDA

$env:NNUNET_RAW = $NnUNetRaw
$env:NNUNET_PREPROCESSED = $NnUNetPreprocessed
$env:NNUNET_RESULTS = $NnUNetResults

Write-Host "NNUNET_RAW=$($env:NNUNET_RAW)"
Write-Host "NNUNET_PREPROCESSED=$($env:NNUNET_PREPROCESSED)"
Write-Host "NNUNET_RESULTS=$($env:NNUNET_RESULTS)"
Write-Host "nnUNet_n_proc_DA=$($env:nnUNet_n_proc_DA)"
Write-Host "OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
Write-Host "OPENBLAS_NUM_THREADS=$($env:OPENBLAS_NUM_THREADS)"
Write-Host "MKL_NUM_THREADS=$($env:MKL_NUM_THREADS)"
Write-Host "NUMEXPR_NUM_THREADS=$($env:NUMEXPR_NUM_THREADS)"

$cmd = @(
    "nnUNetv2_train",
    "$DatasetId",
    "$Configuration",
    "$Fold",
    "-device",
    "$Device"
)

if ($Npz) {
    $cmd += "--npz"
}

if ($Continue) {
    $cmd += "--c"
}

Write-Host "Running: $($cmd -join ' ')"
if ($DryRun) {
    Write-Host "Dry run enabled. Command not executed."
    exit 0
}

& $cmd[0] $cmd[1..($cmd.Count - 1)]
