
# Semantic Segmentation of Brain Structures from CTA Images Using Deep Learning and NoSQL-based ETL Pipelines



# TopBrain - Impact of Database Integration on 3D Brain Vessel Segmentation (CTA)

This repository presents a research pipeline for semantic segmentation of brain vessels from 3D CTA images using deep learning, with a focus on the impact of integrating a NoSQL database (MongoDB) into the data and model pipeline. The project compares several model architectures (including hierarchical segmentation) and demonstrates that leveraging a structured database for data management and ETL (Extract, Transform, Load) processes can improve the performance and reproducibility of AI models in medical imaging.

**Key points:**
- The main research question is: _Does integrating a database into the AI pipeline improve segmentation results for brain vessels?_
- Multiple model architectures are compared (hierarchical segmentation is just one of them).
- The addition of a database (MongoDB) for data storage, annotation, and ETL automation leads to better data quality, experiment tracking, and improved model results.
- Despite these improvements, the detection of ultra-thin vessels remains a major challenge and is identified as a potential topic for future doctoral research.


## Why This Repo

This repository is designed to:
- Demonstrate the impact of database-driven data engineering and ETL on deep learning for medical image segmentation
- Compare different model architectures (not limited to hierarchical)
- Provide reproducible experiments and clear diagnostics
- Report not only metrics but also the practical limitations and open challenges (notably, the segmentation of ultra-thin vessels)

## Problem

Segmenting brain vessels is difficult because of:

- extreme class imbalance (tiny vessels vs large background)
- high anatomical variability across patients
- thin structures and low local contrast


## Method Overview

The project implements and compares several 3D segmentation models (including SwinUNETR and hierarchical approaches) on brain vessel segmentation tasks. The pipeline is built around:
- Automated ETL flows using MongoDB for data storage, annotation, and experiment management
- Patch-based 3D training (PyTorch/MONAI)
- Foreground oversampling and DiceCE loss


## Current Results (Fold 1)

_Results will be updated with final graphs and metrics. See diagnostic JSON files for current experiment outputs._


## Visual Outputs

_Final plots and illustrations will be added soon._

## Reproducibility

### 1) Environment

```powershell
python -m venv env_gpu
env_gpu\Scripts\activate
pip install -r requirements.txt
```

### 2) Data & DB setup

- Configure MongoDB credentials in `.env` (see `.env.example`).
- Main DB used in experiments: `TopBrain_DB`.

### 3) Training commands

Stage 1:

```powershell
env_gpu/Scripts/python.exe 4_Unet3D/train_unet3d_binary.py --collection "MultiClassPatients3D_Binary_CTA41" --target-size "128x128x64" --partition-file "3_Data_Partitionement/partition_materialized.json" --num-classes 2 --fold fold_1 --epochs 300 --patch-size 64 64 64 --swin-feature-size 24 --batch-size 1 --accum-steps 8 --patches-per-volume 12 --train-fg-oversample-prob 0.90 --loss dicece --lambda-dice 2.0 --lambda-ce 0.5 --class-weights "0.05,1.0" --lr 3e-4 --augment --amp --early-stopping 50 --max-hours 12 --save-dir "4_Unet3D/checkpoints/stage1_binary_v2"
```

Stage 2:

```powershell
env_gpu/Scripts/python.exe 5_HierarchicalSeg/level1_families/train_level1.py --collection "HierarchicalPatients3D_Level1_CTA41" --target-size "128x128x64" --partition-file "3_Data_Partitionement/partition_materialized.json" --num-classes 5 --fold fold_1 --epochs 300 --patch-size 64 64 64 --swin-feature-size 24 --batch-size 1 --accum-steps 8 --patches-per-volume 12 --train-fg-oversample-prob 0.90 --loss dicece --lambda-dice 2.0 --lambda-ce 0.5 --class-weights "0.050,1.388,1.031,0.971,0.527" --lr 3e-4 --augment --amp --early-stopping 50 --max-hours 12 --init-checkpoint "4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth" --log-label-distribution --log-foreground-ratio --save-dir "5_HierarchicalSeg/checkpoints/stage2_level1_v1"
```

Stage 3:

```powershell
env_gpu/Scripts/python.exe 5_HierarchicalSeg/level2_fine/train_level2.py --collection "HierarchicalPatients3D_Level2_CTA41_fold1" --target-size "128x128x64" --partition-file "3_Data_Partitionement/partition_materialized.json" --fold fold_1 --num-classes 41 --epochs 300 --patch-size 64 64 64 --swin-feature-size 24 --batch-size 1 --accum-steps 8 --patches-per-volume 12 --train-fg-oversample-prob 0.90 --loss dicece --lambda-dice 2.0 --lambda-ce 0.5 --auto-class-weights --lr 3e-4 --augment --amp --early-stopping 50 --max-hours 12 --init-checkpoint "5_HierarchicalSeg/checkpoints/stage2_level1_v1/swinunetr_level1_best_fold_1.pth" --save-dir "5_HierarchicalSeg/checkpoints/stage3_level2_v1"
```

## Repository Layout

```text
TopBrain_Project/
|- 1_ETL/                      # Extract/Transform/Load pipeline
|- 2_data_augmentation/        # MONAI-based augmentation utilities
|- 3_Data_Partitionement/      # Fold definitions and split materialization
|- 4_Unet3D/                   # Stage-1 training + core 3D model scripts
|- 5_HierarchicalSeg/
|  |- level1_families/         # Stage-2 family-level segmentation
|  |- level2_fine/             # Stage-3 fine-grained segmentation
|  |- checkpoints/
|- diagnose_stage1_recall.py
|- diagnose_level1_families.py
|- diagnose_level2_fine.py
|- results/                    # Diagnostics and run reports
|- docs/                       # Method notes and experiment documentation
```

## Tech Stack

- Python
- PyTorch
- MONAI
- NumPy / SciPy
- nibabel
- MongoDB + PyMongo
- matplotlib

## Scientific Rigor Notes

- Fixed fold protocol via [3_Data_Partitionement/partition_materialized.json](3_Data_Partitionement/partition_materialized.json)
- Explicit diagnostics scripts for each stage
- Separate evaluation of global foreground performance and clinically relevant classes
- Hierarchical class definition tracked in [class_hierarchy.json](class_hierarchy.json)

## Roadmap

- Run full CV5 on Stage 3 and aggregate confidence intervals
- Improve rare-class sampling and loss reweighting on Level-2
- Add per-class calibration and uncertainty maps
- Package reproducible training/evaluation entry points under `src/`

## Disclaimer

This repository contains research code and experiment artifacts. Dataset access may be restricted depending on licensing and institutional rules.