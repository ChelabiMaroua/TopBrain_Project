# UNet2D + UNet3D (500 epochs) — GPU Run Guide for Coworker

## Goal

Run a full comparison for thesis:

1. UNet2D with 3 strategies: directfiles, binary, polygons
2. UNet3D with 3 strategies: directfiles, binary, polygons
3. Automatic final comparison report (best model/strategy)

All runs must use 500 epochs.

## Files to use

- Main orchestrator script: `run_compare_500epochs_gpu.ps1`
- 2D trainer: `4_Unet2D/train_unet2d_compare.py`
- 3D trainer: `4_Unet3D/train_unet3d_compare.py`
- Auto-comparison report: `compare_unet2d_unet3d.py`

## Prerequisites

1. GPU machine with CUDA-compatible PyTorch
2. MongoDB running
3. Partition file exists: `3_Data_Partitionement/partition_materialized.json`
4. 2D ETL already populated (binary + polygons)
5. 3D collections available:
   - binary: `MultiClassPatients`
   - polygons: `PolygonPatients`
6. Python env includes: `torch`, `numpy`, `nibabel`, `opencv-python`, `pymongo`, `python-dotenv`, `monai`

## Recommended command (PowerShell)

From project root:

```powershell
cd C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Project

./run_compare_500epochs_gpu.ps1 `
  -PythonExe "python" `
  -Fold "fold_1" `
  -BatchSize2D 8 `
  -BatchSize3D 1 `
  -NumWorkers 4 `
  -GpuId "0" `
  -RunEtl2D
```

This script now contains all commands explicitly:

1. Optional ETL 2D population (`1_ETL/Load/load_t6_mongodb_insert_2d.py`)
2. UNet2D train with 3 strategies for 500 epochs
3. UNet3D train with 3 strategies for 500 epochs
4. Final 2D vs 3D comparison report generation

It also writes step logs and marker files:

- Logs: `results/run_logs/`
- Resume markers: `results/run_markers/`

## Resume after interruption (important)

If machine stops in the middle, rerun with `-Resume`.
Completed steps are skipped automatically.

```powershell
./run_compare_500epochs_gpu.ps1 `
  -PythonExe "python" `
  -Fold "fold_1" `
  -BatchSize2D 8 `
  -BatchSize3D 1 `
  -NumWorkers 4 `
  -GpuId "0" `
  -Resume
```

If needed, run only missing part:

```powershell
# Skip 2D, rerun 3D + compare
./run_compare_500epochs_gpu.ps1 -Skip2D -Resume

# Skip both training blocks, regenerate only comparison file
./run_compare_500epochs_gpu.ps1 -Skip2D -Skip3D
```

## Prevent 40-minute stop (Windows power)

Before long training, keep laptop plugged in and disable sleep on AC:

```powershell
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
```

After experiment, you can restore your preferred values.

## Equivalent one-liner (without .ps1)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; python run_unet2d_3d_compare.py --fold fold_1 --epochs-2d 500 --epochs-3d 500 --batch-size-2d 8 --batch-size-3d 1 --num-workers 4 --sampling-mode-2d class-aware --sampling-mode-3d class-aware --foreground-boost-2d 2.5 --foreground-boost-3d 2.0 --class-boosts-2d "3:5.0,5:7.0,4:2.0" --class-boosts-3d "3:5.0,5:7.0,4:2.0" --max-sample-weight-2d 14 --max-sample-weight-3d 12
```

## Expected outputs

- 2D training JSON: `results/unet2d_train_results.json`
- 3D training JSON: `results/unet3d_train_results.json`
- Auto summary JSON: `results/unet_compare_summary.json`
- Auto summary Markdown: `results/unet_compare_summary.md`
- Best checkpoints:
  - `4_Unet2D/checkpoints/unet2d_best_directfiles_fold_1.pth`
  - `4_Unet2D/checkpoints/unet2d_best_binary_fold_1.pth`
  - `4_Unet2D/checkpoints/unet2d_best_polygons_fold_1.pth`
  - `4_Unet3D/checkpoints/unet3d_best_directfiles_fold_1.pth`
  - `4_Unet3D/checkpoints/unet3d_best_binary_fold_1.pth`
  - `4_Unet3D/checkpoints/unet3d_best_polygons_fold_1.pth`

## What to send back after run

Commit/push these files:

1. `results/unet2d_train_results.json`
2. `results/unet3d_train_results.json`
3. `results/unet_compare_summary.json`
4. `results/unet_compare_summary.md`

Optional (for plots/reporting):

1. `results/plots/combined_curves_fold_1.png`
2. `results/plots/dice_curves_fold_1.png`
3. `results/plots/iou_curves_fold_1.png`
4. `results/plots/loss_curves_fold_1.png`
5. `results/plots/summary_stats_fold_1.png`

## Troubleshooting

### CUDA out of memory

- Set `-BatchSize3D 1` (already recommended)
- Reduce workers to `-NumWorkers 2`
- If needed lower 2D batch size to 4

### Empty dataset for binary/polygons

- Ensure MongoDB has required collections and target size
- Re-run ETL loaders before training

### MONAI not installed

```powershell
pip install monai
```

### Mongo connection refused

- Start MongoDB service
- Verify `MONGO_URI` and `MONGO_DB_NAME`

## Final note for thesis workflow

After run completion, use `results/unet_compare_summary.md` as the base table for model selection and conclusion.
