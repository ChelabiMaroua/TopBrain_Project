# UNet3D - Mode Binary Mongo

## Statut actuel

- Collection `MultiClassPatients3D_Binary_CTA41`: OK (25 docs, target_size `128x128x64`)

---

## 1) Environnement

```powershell
Set-Location C:/Users/LENOVO/Desktop/PFE-APP/TopBrain_Project
.\.venv-1\Scripts\Activate.ps1
```

---

## 2) Variables (session)

```powershell
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DB_NAME = "TopBrain_DB"
$env:TOPBRAIN_PARTITION_FILE = "3_Data_Partitionement/partition_materialized.json"
```

---

## 3) Vérif rapide collection

```powershell
python -c "from pymongo import MongoClient; c=MongoClient('mongodb://localhost:27017'); col=c['TopBrain_DB']['MultiClassPatients3D_Binary_CTA41']; print(col.count_documents({}), col.distinct('target_size')); c.close()"
```

---

## 4) Lancer UNet3D binary

```powershell
python 4_Unet3D/train_unet3d_compare.py `
  --strategy binary `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --binary-collection MultiClassPatients3D_Binary_CTA41 `
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

## 5) Résultats

- JSON: `results/unet3d_train_results.json`
- Checkpoints: `4_Unet3D/checkpoints/`
