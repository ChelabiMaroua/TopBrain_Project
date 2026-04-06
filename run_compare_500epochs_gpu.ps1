param(
    [string]$PythonExe = "python",
    [string]$Fold = "fold_1",
    [int]$BatchSize2D = 8,
    [int]$BatchSize3D = 1,
    [int]$NumWorkers = 4,
    [string]$GpuId = "0"
)

$ErrorActionPreference = "Stop"

Write-Host "Using Python: $PythonExe"
Write-Host "Fold: $Fold"
Write-Host "2D epochs: 500 | 3D epochs: 500"
Write-Host "Batch sizes -> 2D: $BatchSize2D | 3D: $BatchSize3D"
Write-Host "Num workers: $NumWorkers"

# Pin run to one GPU (edit/remove if multi-GPU scheduling differs)
$env:CUDA_VISIBLE_DEVICES = $GpuId

& $PythonExe run_unet2d_3d_compare.py `
    --fold $Fold `
    --epochs-2d 500 `
    --epochs-3d 500 `
    --batch-size-2d $BatchSize2D `
    --batch-size-3d $BatchSize3D `
    --num-workers $NumWorkers

Write-Host "Done."
Write-Host "2D results: results/unet2d_train_results.json"
Write-Host "3D results: results/unet3d_train_results.json"
Write-Host "Comparison: results/unet_compare_summary.md"
