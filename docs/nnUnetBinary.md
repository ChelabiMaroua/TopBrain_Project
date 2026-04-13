# nnUnetBinary (Splits guidés par MongoDB)

## Objectif

Utiliser nnUNet 3D classique, mais avec un `splits_final.json` construit à partir des scores de richesse en classes rares issus de MongoDB (`mongo_split.py`).

---

## 0) Pré-requis

- Même environnement que `nnUNet_Classic.md`
- Collections multiclasses déjà créées (sinon refaire les étapes ETL ci-dessous)

---

## 1) Cloner + environnement (nouvelle machine)

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

## 3) Peupler la BDD MongoDB (multiclasse)

### 3.1 ETL 3D multiclasse (T5)

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

### 3.2 ETL 2D multiclasse (T6) — utilisé pour scorer Mongo

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

## 4) Partition de base + dataset nnUNet

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

## 5) Créer les splits MongoDB enrichis

```powershell
python 5_nnUNet/splits/mongo_split.py `
  --partition-file "$env:TOPBRAIN_PARTITION_FILE" `
  --collection MultiClassPatients2D_Binary_CTA41 `
  --target-size 128x128x64 `
  --dataset-id 501 `
  --dataset-name TopBrainCTA `
  --output-report results/mongo_split_report.json
```

Fichier produit:
- `nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final_MONGO.json`

---

## 6) Sauvegarder les splits random puis activer les splits Mongo

```powershell
Copy-Item "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final.json" "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final_RANDOM.json" -Force
Copy-Item "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final_MONGO.json" "nnUNet_preprocessed/Dataset501_TopBrainCTA/splits_final.json" -Force
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

## 8) Lancer l'entraînement nnUNetBinary (splits Mongo)

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

---

## 10) Comparaison A/B recommandée

- Run A: `splits_final_RANDOM.json`
- Run B: `splits_final_MONGO.json`
- Comparer Dice/IoU fold par fold sur les mêmes folds 0..4
