# nnUnetPolygones (nnUNet avec splits guidés par PolygonMongo)

## Objectif

Utiliser nnUNet 3D (`Dataset501_TopBrainCTA`) avec des splits construits depuis la collection:
- `MultiClassPatients2D_Polygons_CTA41`

Le scoring est topologique (contours polygonaux par classe), puis conversion en `splits_final.json` pour nnUNet.

---

## 1) Pré-requis (nouvelle machine GPU)

```powershell
git clone https://github.com/ChelabiMaroua/TopBrain_Project.git
Set-Location TopBrain_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pymongo python-dotenv nibabel opencv-python monai
powershell -ExecutionPolicy Bypass -File 5_nnUNet/setup/setup_nnunet_cuda_4060.ps1 -PythonCmd python
```

---

## 2) Variables d'environnement (session)

```powershell
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DB_NAME = "TopBrain_DB"

$env:TOPBRAIN_IMAGE_DIR = "C:/DATA/TopBrain/imagesTr_topbrain_ct"
$env:TOPBRAIN_LABEL_DIR = "C:/DATA/TopBrain/labelsTr_topbrain_ct"

$env:NNUNET_RAW = "nnUNet_raw"
$env:NNUNET_PREPROCESSED = "nnUNet_preprocessed"
$env:NNUNET_RESULTS = "nnUNet_results"

$env:NNUNET_DATASET_ID = "501"
$env:NNUNET_DATASET_NAME = "TopBrainCTA"
$env:TOPBRAIN_PARTITION_FILE = "3_Data_Partitionement/partition_materialized.json"
```

---

## 3) Peupler MongoDB (multiclasse)

### 3.1 T5 (3D multiclasse)

```powershell
python 1_ETL/Load/load_t5_mongodb_insert.py `
  --image-dir "$env:TOPBRAIN_IMAGE_DIR" `
  --label-dir "$env:TOPBRAIN_LABEL_DIR" `
  --target-size 128 128 64 `
  --class-min 0 `
  --class-max 40 `
  --window-min 0 `
  --window-max 600 `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --collection MultiClassPatients3D_Binary_CTA41 `
  --keep-multiclass-labels
```

### 3.2 T6 (2D binary + polygons multiclasse)

```powershell
python 1_ETL/Load/load_t6_mongodb_insert_2d.py `
  --image-dir "$env:TOPBRAIN_IMAGE_DIR" `
  --label-dir "$env:TOPBRAIN_LABEL_DIR" `
  --target-size 128 128 64 `
  --class-min 0 `
  --class-max 40 `
  --num-classes 41 `
  --window-min 0 `
  --window-max 600 `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --binary-collection MultiClassPatients2D_Binary_CTA41 `
  --polygon-collection MultiClassPatients2D_Polygons_CTA41
```

---

## 4) Préparer dataset nnUNet et partition de base

```powershell
python 3_Data_Partitionement/partition_data.py `
  --mongo-uri "$env:MONGO_URI" `
  --db-name "$env:MONGO_DB_NAME" `
  --collection MultiClassPatients3D_Binary_CTA41 `
  --k 5 `
  --test-ratio 0.2 `
  --seed 42 `
  --output "$env:TOPBRAIN_PARTITION_FILE"

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

## 5) Générer les splits PolygonMongo

```powershell
python 5_nnUNet/splits/mongo_split_polygons.py `
  --partition-file "$env:TOPBRAIN_PARTITION_FILE" `
  --collection MultiClassPatients2D_Polygons_CTA41 `
  --target-size 128x128x64 `
  --dataset-id 501 `
  --dataset-name TopBrainCTA `
  --output-report results/mongo_split_polygons_report.json
```

Fichiers générés:
- `nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final_MONGO_POLYGONS.json`
- `results/mongo_split_polygons_report.json`

---

## 6) Activer les splits PolygonMongo dans nnUNet

```powershell
Copy-Item "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final_MONGO_POLYGONS.json" "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final.json" -Force
```

---

## 7) Plan + preprocess

```powershell
powershell -ExecutionPolicy Bypass -File 5_nnUNet/setup/run_plan_preprocess_lowmem.ps1 `
  -DatasetId 501 `
  -Configs 3d_fullres `
  -NnUNetRaw "$env:NNUNET_RAW" `
  -NnUNetPreprocessed "$env:NNUNET_PREPROCESSED" `
  -NnUNetResults "$env:NNUNET_RESULTS"
```

---

## 8) Entraînement nnUNet (5 folds)

```powershell
nnUNetv2_train 501 3d_fullres 0 -device cuda --npz
nnUNetv2_train 501 3d_fullres 1 -device cuda --npz
nnUNetv2_train 501 3d_fullres 2 -device cuda --npz
nnUNetv2_train 501 3d_fullres 3 -device cuda --npz
nnUNetv2_train 501 3d_fullres 4 -device cuda --npz
```

---

## 9) Reprise en cas d'arrêt

```powershell
nnUNetv2_train 501 3d_fullres 0 -device cuda --npz --c
```
