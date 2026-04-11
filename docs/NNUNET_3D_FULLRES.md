# nnUNet 3D Fullres - Implementation Guide (TopBrain CTA41)

This project now includes a complete nnUNet v2 3D fullres integration in `5_nnUNet/`.

## Added scripts

- `5_nnUNet/prepare_nnunet_dataset.py`
  - Prepares `nnUNet_raw/DatasetXXX_*` from TopBrain image/label folders.
  - Generates `dataset.json` with 41 labels (0..40).
- `5_nnUNet/create_nnunet_splits.py`
  - Converts `partition_materialized.json` into `splits_final.json` for nnUNet v2.

## 1) Install nnUNet v2

```bash
pip install nnunetv2
```

## 2) Set nnUNet env vars

Windows PowerShell:

```powershell
$env:NNUNET_RAW="nnUNet_raw"
$env:NNUNET_PREPROCESSED="nnUNet_preprocessed"
$env:NNUNET_RESULTS="nnUNet_results"
```

Linux/Colab:

```bash
export NNUNET_RAW=nnUNet_raw
export NNUNET_PREPROCESSED=nnUNet_preprocessed
export NNUNET_RESULTS=nnUNet_results
```

## 3) Prepare nnUNet dataset (raw)

```bash
python 5_nnUNet/prepare_nnunet_dataset.py \
  --image-dir /path/to/imagesTr_topbrain_ct \
  --label-dir /path/to/labelsTr_topbrain_ct \
  --nnunet-raw nnUNet_raw \
  --dataset-id 501 \
  --dataset-name TopBrainCTA \
  --mode copy
```

Result:

- `nnUNet_raw/Dataset501_TopBrainCTA/imagesTr/*_0000.nii.gz`
- `nnUNet_raw/Dataset501_TopBrainCTA/labelsTr/*.nii.gz`
- `nnUNet_raw/Dataset501_TopBrainCTA/dataset.json`

## 4) Create custom K-fold splits for nnUNet

```bash
python 5_nnUNet/create_nnunet_splits.py \
  --partition-file 3_Data_Partitionement/partition_materialized.json \
  --nnunet-preprocessed nnUNet_preprocessed \
  --dataset-id 501 \
  --dataset-name TopBrainCTA
```

Result:

- `nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final.json`

## 5) Plan + preprocess

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

Windows low-memory safe variant (recommended if you see paging-file or DLL load errors):

```powershell
.\5_nnUNet\setup\run_plan_preprocess_lowmem.ps1 -DatasetId 501
```

Equivalent manual command:

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -npfp 1 -np 1
```

If you still get `The paging file is too small for this operation to complete`:

- Increase Windows virtual memory (paging file) and reboot.
- Keep only one preprocessing process (`-npfp 1 -np 1`) and avoid running other heavy jobs.
- Run only `3d_fullres` first (as shown above) before adding other configs.

## 6) Train 3D fullres

Train fold 0:

```bash
nnUNetv2_train 501 3d_fullres 0 --npz
```

Windows low-memory safe variant:

```powershell
.\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId 501 -Configuration 3d_fullres -Fold 0 -Device cuda -NProcDA 2 -Npz
```

Preview command only (no execution):

```powershell
.\5_nnUNet\setup\run_train_lowmem.ps1 -DatasetId 501 -Configuration 3d_fullres -Fold 0 -DryRun
```

Notes for low-memory runs:

- Reduce `-NProcDA` to `1` if you still see memory pressure.
- Use `-Device cpu` only if GPU is unavailable (slower, typically more RAM usage on host).
- Resume interrupted training with `-Continue`.

Train all folds (0..4):

```bash
nnUNetv2_train 501 3d_fullres 0 --npz
nnUNetv2_train 501 3d_fullres 1 --npz
nnUNetv2_train 501 3d_fullres 2 --npz
nnUNetv2_train 501 3d_fullres 3 --npz
nnUNetv2_train 501 3d_fullres 4 --npz
```

## 7) Predict (example)

```bash
nnUNetv2_predict \
  -i nnUNet_raw/Dataset501_TopBrainCTA/imagesTr \
  -o nnUNet_predictions/fold0 \
  -d 501 \
  -c 3d_fullres \
  -f 0
```

## Notes

- This is an implementation layer; it does not modify your UNet2D/UNet3D custom trainers.
- For TopBrain CTA, labels are full multiclass (41 classes).
- If disk space is limited, use `--mode hardlink` in `prepare_nnunet_dataset.py`.
