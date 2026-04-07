# GPU Run Commands

## Status

The core 2D and 3D training scripts are present and runnable for the main compare workflow.
The repository still has cleanup deletions in the worktree, but the execution path below is the one to use on the GPU machine.

## Prerequisites

- MongoDB running
- `3_Data_Partitionement/partition_materialized.json` available
- `1_ETL/` already completed
- 2D Mongo collections available:
  - `MultiClassPatients2D_Binary`
  - `MultiClassPatients2D_Polygons`
- Python environment with: `torch`, `numpy`, `nibabel`, `opencv-python`, `pymongo`, `python-dotenv`, `monai`

## Main GPU command

From the project root:

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

This runs:

1. Optional ETL load for the 2D Mongo collections
2. UNet2D training on the 3 strategies for 500 epochs
3. UNet3D training on the 3 strategies for 500 epochs
4. Final 2D vs 3D comparison report

## Resume after interruption

```powershell
cd C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Project

./run_compare_500epochs_gpu.ps1 `
  -PythonExe "python" `
  -Fold "fold_1" `
  -BatchSize2D 8 `
  -BatchSize3D 1 `
  -NumWorkers 4 `
  -GpuId "0" `
  -Resume
```

## Run only the missing parts

```powershell
# Skip 2D, rerun 3D + comparison
./run_compare_500epochs_gpu.ps1 -Skip2D -Resume

# Skip both trainings, rebuild only comparison report
./run_compare_500epochs_gpu.ps1 -Skip2D -Skip3D
```

## Direct commands, if you need to run each step manually

### 2D

```powershell
python 4_Unet2D/train_unet2d_compare.py `
  --strategy all `
  --fold fold_1 `
  --epochs 500 `
  --batch-size 8 `
  --num-workers 4
```

### 3D

```powershell
python 4_Unet3D/train_unet3d_compare.py `
  --strategy all `
  --fold fold_1 `
  --epochs 500 `
  --batch-size 1 `
  --num-workers 4
```

### Comparison report only

```powershell
python compare_unet2d_unet3d.py `
  --unet2d-json results/unet2d_train_results.json `
  --unet3d-json results/unet3d_train_results.json `
  --output-json results/unet_compare_summary.json `
  --output-md results/unet_compare_summary.md
```

## Expected outputs

- `results/unet2d_train_results.json`
- `results/unet3d_train_results.json`
- `results/unet_compare_summary.json`
- `results/unet_compare_summary.md`
- `results/run_logs/`
- `results/run_markers/`

## Power settings

If the GPU machine is a Windows laptop, disable sleep during the run:

```powershell
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
```

## Troubleshooting

- If `MongoDB` fails, check the service is running and `MONGO_URI` is correct.
- If a dataset is empty, verify the partition file and the 2D collections.
- If CUDA runs out of memory, reduce `-BatchSize3D` to `1` and then reduce `-BatchSize2D` to `4`.
- If `MONAI` is missing, install it with `pip install monai`.
