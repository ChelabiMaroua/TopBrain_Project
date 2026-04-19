$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = "c:/Users/kirua/PFEMaroua/.venv/Scripts/python.exe"
$imageDir = "c:/Users/kirua/PFEMaroua/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct"
$labelDir = "c:/Users/kirua/PFEMaroua/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct"
$partition = "3_Data_Partitionement/partition_materialized.json"

$env:CUDA_VISIBLE_DEVICES = '0'

# Fold 1
& $python 4_Unet2D/train_unet2d_compare.py `
  --strategy directfiles `
  --image-dir $imageDir `
  --label-dir $labelDir `
  --partition-file $partition `
  --fold fold_1 `
  --target-size 256 256 64 `
  --require-cuda `
  --epochs 200 `
  --batch-size 2 `
  --grad-accum-steps 2 `
  --base-channels 32 `
  --num-workers 0 `
  --lr 2e-4 `
  --loss ce `
  --background-weight-scale 0.10 `
  --eta-min-lr 1e-6 `
  --sampling-mode class-aware `
  --foreground-boost 10 `
  --train-patch-size 192 `
  --fg-center-prob 0.85 `
  --ema-decay 0 `
  --early-stopping 35 `
  --min-epochs 45 `
  --output-json "results/unet2d_directfiles_fold_1_final.json"

# Fold 5
& $python 4_Unet2D/train_unet2d_compare.py `
  --strategy directfiles `
  --image-dir $imageDir `
  --label-dir $labelDir `
  --partition-file $partition `
  --fold fold_5 `
  --target-size 256 256 64 `
  --require-cuda `
  --epochs 200 `
  --batch-size 2 `
  --grad-accum-steps 2 `
  --base-channels 32 `
  --num-workers 0 `
  --lr 2e-4 `
  --loss ce `
  --background-weight-scale 0.10 `
  --eta-min-lr 1e-6 `
  --sampling-mode class-aware `
  --foreground-boost 10 `
  --train-patch-size 192 `
  --fg-center-prob 0.85 `
  --ema-decay 0 `
  --early-stopping 35 `
  --min-epochs 45 `
  --output-json "results/unet2d_directfiles_fold_5_final.json"

Write-Host "Done. Check results/" -ForegroundColor Green