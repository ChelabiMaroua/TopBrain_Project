param(
    [int]$DatasetId = 501,
    [string]$Configs = "3d_fullres",
    [string]$NnUNetRaw = "nnUNet_raw",
    [string]$NnUNetPreprocessed = "nnUNet_preprocessed",
    [string]$NnUNetResults = "nnUNet_results"
)

$ErrorActionPreference = "Stop"

# Keep OpenMP/BLAS thread counts low to prevent memory spikes during worker imports.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$env:NNUNET_RAW = $NnUNetRaw
$env:NNUNET_PREPROCESSED = $NnUNetPreprocessed
$env:NNUNET_RESULTS = $NnUNetResults

Write-Host "NNUNET_RAW=$($env:NNUNET_RAW)"
Write-Host "NNUNET_PREPROCESSED=$($env:NNUNET_PREPROCESSED)"
Write-Host "NNUNET_RESULTS=$($env:NNUNET_RESULTS)"
Write-Host "OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
Write-Host "OPENBLAS_NUM_THREADS=$($env:OPENBLAS_NUM_THREADS)"
Write-Host "MKL_NUM_THREADS=$($env:MKL_NUM_THREADS)"
Write-Host "NUMEXPR_NUM_THREADS=$($env:NUMEXPR_NUM_THREADS)"

# Use 1 process for fingerprinting and preprocessing to avoid paging-file crashes on Windows.
nnUNetv2_plan_and_preprocess -d $DatasetId --verify_dataset_integrity -c $Configs -npfp 1 -np 1 --verbose
