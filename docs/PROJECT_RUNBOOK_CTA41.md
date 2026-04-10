# TopBrain Project - Guide Projet et Runbook (CTA41)

## 1) Objectif du projet

Ce projet implemente un pipeline complet de segmentation multi-classes des vaisseaux cerebraux TopBrain:

- ETL (chargement et structuration des donnees)
- augmentation
- partitionnement K-Fold
- entrainement UNet2D / UNet3D
- comparaison et export des resultats

Le mode actuel est aligne sur CTA41:

- classes: 0..40 (41 classes)
- fenetre CT: [0, 600]
- collections MongoDB dediees CTA41

## 2) Structure du projet

- `1_ETL/`
  - `Extract/`: detection et listing des fichiers patient
  - `Transform/`: cast, resize, normalisation, serialisation
  - `Load/`: insertion MongoDB (2D binary/polygons, 3D binary/polygons)
- `2_data_augmentation/`
  - pipeline MONAI + visualisation de controle
- `3_Data_Partitionement/`
  - creation du fichier de folds train/val/test depuis MongoDB
- `4_Unet2D/`
  - modele UNet2D + script d'entrainement comparatif
- `4_Unet3D/`
  - modele UNet3D + scripts d'entrainement
- `docs/`
  - documentation, commandes, runbooks
- `results/`
  - sorties d'entrainement (json, logs, metriques)

## 3) Variables d'environnement attendues

Le fichier `.env` doit pointer vers:

- dossier images et labels TopBrain CTA
- MongoDB locale
- collections CTA41

Valeurs importantes deja alignees dans ce projet:

- `TOPBRAIN_NUM_CLASSES=41`
- `TOPBRAIN_CTA_WINDOW_MIN=0`
- `TOPBRAIN_CTA_WINDOW_MAX=600`
- `TOPBRAIN_2D_BINARY_COLLECTION=MultiClassPatients2D_Binary_CTA41`
- `TOPBRAIN_2D_POLYGON_COLLECTION=MultiClassPatients2D_Polygons_CTA41`
- `TOPBRAIN_3D_BINARY_COLLECTION=MultiClassPatients3D_Binary_CTA41`
- `TOPBRAIN_3D_POLYGON_COLLECTION=MultiClassPatients3D_Polygons_CTA41`

## 4) Ordre d'execution recommande

## Etape A - Creer la BDD et les collections avec ETL

Lancer depuis la racine du projet.

### A1. ETL 3D Binary (T5)

```powershell
D:/Python314/python.exe 1_ETL/Load/load_t5_mongodb_insert.py `
  --image-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct" `
  --label-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct" `
  --target-size 128 128 64 `
  --class-min 0 `
  --class-max 40 `
  --window-min 0 `
  --window-max 600 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --collection MultiClassPatients3D_Binary_CTA41 `
  --keep-multiclass-labels
```

### A2. ETL 3D Polygons (T7)

```powershell
D:/Python314/python.exe 1_ETL/Load/load_t7_mongodb_insert_3d_polygons.py `
  --image-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct" `
  --label-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct" `
  --target-size 128 128 64 `
  --class-min 0 `
  --class-max 40 `
  --num-classes 41 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --collection MultiClassPatients3D_Polygons_CTA41
```

### A3. ETL 2D Binary + Polygons (T6)

```powershell
D:/Python314/python.exe 1_ETL/Load/load_t6_mongodb_insert_2d.py `
  --image-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct" `
  --label-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct" `
  --target-size 256 256 192 `
  --class-min 0 `
  --class-max 40 `
  --num-classes 41 `
  --window-min 0 `
  --window-max 600 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --binary-collection MultiClassPatients2D_Binary_CTA41 `
  --polygon-collection MultiClassPatients2D_Polygons_CTA41
```

## Etape B - Data Augmentation (controle)

Le pipeline d'augmentation est applique pendant l'entrainement.

Pour verifier visuellement les transformations:

```powershell
D:/Python314/python.exe 2_data_augmentation/visualize_patient_monai.py `
  --patient-id 001 `
  --target-size 128 128 64 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --collection MultiClassPatients3D_Binary_CTA41 `
  --axis 2 `
  --seed 42
```

## Etape C - Data Partitionement

Generer la partition K-Fold depuis la collection 3D binary:

```powershell
D:/Python314/python.exe 3_Data_Partitionement/partition_data.py `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --collection MultiClassPatients3D_Binary_CTA41 `
  --k 5 `
  --test-ratio 0.2 `
  --seed 42 `
  --output 3_Data_Partitionement/partition_materialized.json
```

## Etape D - Entrainement UNet2D 250 epochs

## D1. Strategy directfiles

```powershell
D:/Python314/python.exe 4_Unet2D/train_unet2d_compare.py `
  --strategy directfiles `
  --image-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct" `
  --label-dir "C:/Users/LENOVO/Desktop/PFFECerine/TopBrain_Data_Release_Batches1n2_081425/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct" `
  --partition-file 3_Data_Partitionement/partition_materialized.json `
  --fold fold_1 `
  --epochs 250 `
  --batch-size 4 `
  --lr 3e-4 `
  --eta-min-lr 1e-6 `
  --base-channels 32 `
  --num-classes 41 `
  --sampling-mode class-aware `
  --foreground-boost 6.0 `
  --class-boosts "" `
  --max-sample-weight 20.0 `
  --early-stopping 40 `
  --min-epochs 60 `
  --num-workers 2
```

## D2. Strategy binary

```powershell
D:/Python314/python.exe 4_Unet2D/train_unet2d_compare.py `
  --strategy binary `
  --partition-file 3_Data_Partitionement/partition_materialized.json `
  --fold fold_1 `
  --epochs 250 `
  --batch-size 4 `
  --lr 3e-4 `
  --eta-min-lr 1e-6 `
  --base-channels 32 `
  --num-classes 41 `
  --sampling-mode class-aware `
  --foreground-boost 6.0 `
  --class-boosts "" `
  --max-sample-weight 20.0 `
  --early-stopping 40 `
  --min-epochs 60 `
  --num-workers 2 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --binary-collection MultiClassPatients2D_Binary_CTA41
```

## D3. Strategy polygons

```powershell
D:/Python314/python.exe 4_Unet2D/train_unet2d_compare.py `
  --strategy polygons `
  --partition-file 3_Data_Partitionement/partition_materialized.json `
  --fold fold_1 `
  --epochs 250 `
  --batch-size 4 `
  --lr 3e-4 `
  --eta-min-lr 1e-6 `
  --base-channels 32 `
  --num-classes 41 `
  --sampling-mode class-aware `
  --foreground-boost 6.0 `
  --class-boosts "" `
  --max-sample-weight 20.0 `
  --early-stopping 40 `
  --min-epochs 60 `
  --num-workers 2 `
  --mongo-uri mongodb://localhost:27017 `
  --db-name TopBrain_DB `
  --polygon-collection MultiClassPatients2D_Polygons_CTA41
```

## Etape E - Lancer UNet2D et UNet3D en une seule commande

Cette commande lance automatiquement:

- UNet2D avec les 3 strategies: `directfiles`, `binary`, `polygons`
- UNet3D avec les 3 strategies: `directfiles`, `binary`, `polygons`

Les resultats sont ensuite compares dans `results/unet_compare_summary.md`.

```powershell
D:/Python314/python.exe run_unet2d_3d_compare.py `
  --fold fold_1 `
  --epochs-2d 250 `
  --epochs-3d 250 `
  --batch-size-2d 4 `
  --batch-size-3d 1 `
  --num-workers 2 `
  --sampling-mode-2d class-aware `
  --sampling-mode-3d class-aware `
  --foreground-boost-2d 6.0 `
  --foreground-boost-3d 2.0 `
  --class-boosts-2d "" `
  --class-boosts-3d "" `
  --max-sample-weight-2d 20.0 `
  --max-sample-weight-3d 12.0
```

## 5) Notes pratiques

- Si execution CPU seulement: reduire `batch-size` (1-2) et `base-channels` (8-16).
- Sur GPU, garder `base-channels=32` pour un compromis vitesse/performance.
- Pour des tests rapides, commencer avec 3 a 10 epochs avant les runs longs.
- Les sorties UNet2D sont ecrites dans `results/unet2d_train_results.json`.
