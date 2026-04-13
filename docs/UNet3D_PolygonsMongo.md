# UNet3D - Mode Polygons Mongo

## Statut actuel

- Collection `MultiClassPatients3D_Polygons_CTA41`: vide (0 docs)
- Il faut d'abord lancer T7.

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
$env:TOPBRAIN_IMAGE_DIR = "C:/Users/LENOVO/Desktop/PFE/data/raw/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct"
$env:TOPBRAIN_LABEL_DIR = "C:/Users/LENOVO/Desktop/PFE/data/raw/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct"
$env:TOPBRAIN_PARTITION_FILE = "3_Data_Partitionement/partition_materialized.json"
```

---

## 3) Peupler la collection 3D Polygons (T7)

```powershell
python 1_ETL/Load/load_t7_mongodb_insert_3d_polygons.py `
  --image-dir "$env:TOPBRAIN_IMAGE_DIR" `
  --label-dir "$env:TOPBRAIN_LABEL_DIR" `
  --target-size 128 128 64 `
  --class-min 0 `
  --class-max 40 `
  --num-classes 41 `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --collection MultiClassPatients3D_Polygons_CTA41
```

---

## 4) Vérifier le peuplement

```powershell
python -c "from pymongo import MongoClient; c=MongoClient('mongodb://localhost:27017'); col=c['TopBrain_DB']['MultiClassPatients3D_Polygons_CTA41']; print(col.count_documents({})); c.close()"
```

---

## 5) Lancer UNet3D polygons

```powershell
python 4_Unet3D/train_unet3d_compare.py `
  --strategy polygons `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --polygon-collection MultiClassPatients3D_Polygons_CTA41 `
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

## 6) Résultats

- JSON: `results/unet3d_train_results.json`
- Checkpoints: `4_Unet3D/checkpoints/`
