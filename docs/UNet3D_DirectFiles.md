# UNet3D - Mode Direct Files

## Statut actuel

- Partition: OK (`3_Data_Partitionement/partition_materialized.json` existe)
- Dossiers images/labels: OK
- Ce mode n'utilise pas MongoDB.

---

## 1) Environnement

```powershell
Set-Location C:/Users/LENOVO/Desktop/PFE-APP/TopBrain_Project
.\.venv-1\Scripts\Activate.ps1
```

---

## 2) Variables (session)

```powershell
$env:TOPBRAIN_IMAGE_DIR = "C:/Users/LENOVO/Desktop/PFE/data/raw/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct"
$env:TOPBRAIN_LABEL_DIR = "C:/Users/LENOVO/Desktop/PFE/data/raw/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct"
$env:TOPBRAIN_PARTITION_FILE = "3_Data_Partitionement/partition_materialized.json"
```

---

## 3) Lancer UNet3D directfiles

```powershell
python 4_Unet3D/train_unet3d_compare.py `
  --strategy directfiles `
  --image-dir "$env:TOPBRAIN_IMAGE_DIR" `
  --label-dir "$env:TOPBRAIN_LABEL_DIR" `
  --partition-file "$env:TOPBRAIN_PARTITION_FILE" `
  --fold fold_1 `
  --target-size 128 128 64 `
  --num-classes 41 `
  --epochs 150 `
  --batch-size 1 `
  --num-workers 0 `
  --sampling-mode class-aware `
  --foreground-boost 2.0 `
  --max-sample-weight 12.0 `
  --early-stopping 20 `
  --min-epochs 40
```

---

## 4) Résultats

- JSON: `results/unet3d_train_results.json`
- Checkpoints: `4_Unet3D/checkpoints/`
