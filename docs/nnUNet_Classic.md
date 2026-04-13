# nnUNet Classic (Nouvelle machine GPU, sans MongoDB)

## 0) Pré-requis

- Windows + GPU NVIDIA (ex: RTX 4060)
- Python 3.10+ (idéalement 3.10/3.11)
- Dataset TopBrain CTA disponible sur disque (images + labels NIfTI)

---

## 1) Cloner le projet et créer l'environnement

```powershell
git clone https://github.com/ChelabiMaroua/TopBrain_Project.git
Set-Location TopBrain_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy python-dotenv nibabel opencv-python monai
```

---

## 2) Installer PyTorch CUDA + nnUNet (profil GPU)

```powershell
powershell -ExecutionPolicy Bypass -File 5_nnUNet/setup/setup_nnunet_cuda_4060.ps1 -PythonCmd python
```

---

## 3) Définir les variables d'environnement (session courante)

```powershell
$env:TOPBRAIN_IMAGE_DIR = "C:/DATA/TopBrain/imagesTr_topbrain_ct"
$env:TOPBRAIN_LABEL_DIR = "C:/DATA/TopBrain/labelsTr_topbrain_ct"

$env:NNUNET_RAW = "nnUNet_raw"
$env:NNUNET_PREPROCESSED = "nnUNet_preprocessed"
$env:NNUNET_RESULTS = "nnUNet_results"

$env:NNUNET_DATASET_ID = "501"
$env:NNUNET_DATASET_NAME = "TopBrainCTA"
```

---

## 4) Préparer dataset nnUNet (NIfTI -> nnUNet_raw)

```powershell
python 5_nnUNet/prepare_nnunet_dataset.py `
  --image-dir "$env:TOPBRAIN_IMAGE_DIR" `
  --label-dir "$env:TOPBRAIN_LABEL_DIR" `
  --nnunet-raw "$env:NNUNET_RAW" `
  --dataset-id 501 `
  --dataset-name TopBrainCTA `
  --mode copy `
  --force
```

---

## 5) Plan + preprocess nnUNet

```powershell
powershell -ExecutionPolicy Bypass -File 5_nnUNet/setup/run_plan_preprocess_lowmem.ps1 `
  -DatasetId 501 `
  -Configs 3d_fullres `
  -NnUNetRaw "$env:NNUNET_RAW" `
  -NnUNetPreprocessed "$env:NNUNET_PREPROCESSED" `
  -NnUNetResults "$env:NNUNET_RESULTS"
```

---

## 6) Entraîner nnUNet Classic (5 folds, splits standards nnUNet)

```powershell
nnUNetv2_train 501 3d_fullres 0 -device cuda --npz
nnUNetv2_train 501 3d_fullres 1 -device cuda --npz
nnUNetv2_train 501 3d_fullres 2 -device cuda --npz
nnUNetv2_train 501 3d_fullres 3 -device cuda --npz
nnUNetv2_train 501 3d_fullres 4 -device cuda --npz
```

---

## 7) Option reprise (si interruption)

```powershell
nnUNetv2_train 501 3d_fullres 0 -device cuda --npz --c
```
