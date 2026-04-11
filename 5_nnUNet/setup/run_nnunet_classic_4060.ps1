param(
    [int]$DatasetId = 1,
    [string]$Configuration = "3d_fullres",
    [int]$Fold = 0,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda",
    [int]$NProcDA = 2,
    [switch]$Npz,
    [switch]$Continue,
    [switch]$SkipPreprocess
)

$ErrorActionPreference = "Stop"

Write-Host "=== nnUNet classic pipeline (RTX 4060 profile) ==="
Write-Host "DatasetId=$DatasetId Configuration=$Configuration Fold=$Fold Device=$Device"

if (-not $SkipPreprocess) {
    Write-Host "[1/2] Running low-memory plan+preprocess..."
    .\5_nnUNet\setup\run_plan_preprocess_lowmem.ps1 -DatasetId $DatasetId
}
else {
    Write-Host "[1/2] Skipped preprocess (SkipPreprocess=true)."
}

Write-Host "[2/2] Running low-memory training..."
if ($Npz -and $Continue) {
    .\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId $DatasetId -Configuration $Configuration -Fold $Fold -Device $Device -NProcDA $NProcDA -Npz -Continue
}
elseif ($Npz) {
    .\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId $DatasetId -Configuration $Configuration -Fold $Fold -Device $Device -NProcDA $NProcDA -Npz
}
elseif ($Continue) {
    .\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId $DatasetId -Configuration $Configuration -Fold $Fold -Device $Device -NProcDA $NProcDA -Continue
}
else {
    .\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId $DatasetId -Configuration $Configuration -Fold $Fold -Device $Device -NProcDA $NProcDA
}
