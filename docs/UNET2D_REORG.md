# UNet2D Reorganization (3D kept as backup)

## Goal
- Make 2D the primary training and comparison path.
- Keep 3D scripts available as fallback/baseline.

## Active 2D Flow (A -> D)
1. Phase A: quick DirectFiles training sanity check.
- Script: 4_Unet2D/train_unet2d_compare.py --strategy directfiles

2. Phase B: one-shot 2D ETL for Mongo Binary + Mongo Polygons.
- Script: 1_ETL/Load/load_t6_mongodb_insert_2d.py
- Collections:
  - TOPBRAIN_2D_BINARY_COLLECTION
  - TOPBRAIN_2D_POLYGON_COLLECTION

3. Phase C: comparative training on 3 strategies up to training stage.
- Script: 4_Unet2D/train_unet2d_compare.py --strategy all
- Strategies:
  - directfiles
  - binary
  - polygons

4. Phase D: KPI2 / KPI3 / KPI4 measurement.
- Script: benchmark_unet2d_kpi.py
- Output JSON: results/unet2d_kpi.json

## One-command runner
- Script: run_unet2d_phases.py
- Example:
  - D:/Python314/python.exe run_unet2d_phases.py --epochs 1 --batch-size 8 --workers 0 1 --fold fold_1

## Kept for backup
- 4_Unet3D/
- Existing 3D scripts are intentionally not deleted.

## Environment variables (2D relevant)
- TOPBRAIN_2D_BINARY_COLLECTION
- TOPBRAIN_2D_POLYGON_COLLECTION
- TOPBRAIN_2D_CHECKPOINT_DIR
- TOPBRAIN_2D_TRAIN_RESULTS_JSON
- TOPBRAIN_PARTITION_FILE
- TOPBRAIN_IMAGE_DIR
- TOPBRAIN_LABEL_DIR
- MONGO_URI
- MONGO_DB_NAME
