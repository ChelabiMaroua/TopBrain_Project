# Coworker GPU CUDA Checklist

## 1) Machine prerequisites
1. Install NVIDIA driver compatible with GPU.
2. Install CUDA toolkit version matching the selected PyTorch build.
3. Verify GPU visibility:
- nvidia-smi

## 2) Python environment
1. Create and activate a clean environment.
2. Install dependencies (PyTorch CUDA, numpy, nibabel, opencv-python, pymongo, monai, python-dotenv).
3. Verify CUDA in Python:
- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

## 3) Configure .env
1. Copy .env.example to .env.
2. Set these values exactly for local machine:
- TOPBRAIN_IMAGE_DIR
- TOPBRAIN_LABEL_DIR
- TOPBRAIN_PARTITION_FILE
- TOPBRAIN_2D_EPOCHS (set to 150)
- TOPBRAIN_2D_BINARY_COLLECTION
- TOPBRAIN_2D_POLYGON_COLLECTION
- MONGO_URI
- MONGO_DB_NAME

## 4) Data and Mongo checks
1. Ensure MongoDB service is running.
2. Quick extract test:
- python 1_ETL/Extract/extract_t0_list_patient_files.py --preview 1
3. Partition test:
- python 3_Data_Partitionement/partition_data.py --k 5 --test-ratio 0.2

## 5) Run full 2D pipeline (A->D)
1. Preferred command:
- python run_unet2d_phases.py --epochs 150 --batch-size 16 --workers 0 2 4 --fold fold_1
2. Expected outputs:
- results/unet2d_train_results.json
- results/unet2d_kpi.json
- results/phase_b_etl_log.txt
3. This run should be executed by the coworker on CUDA to get more precise visuals and more representative throughput numbers.

## 6) GPU-specific verification
1. During Phase C, confirm training uses cuda device in terminal logs.
2. Compare at least two runs:
- batch-size 8 vs 16
- workers 0 vs 4
3. Report if OOM occurs and retry with lower batch size.
4. Keep the 2D pipeline as the main visualization target; the 3D stack remains a backup reference only.

## 7) KPI validation tasks
1. KPI2: check throughput trend by workers for all 3 strategies.
2. KPI3: verify disk occupancy values are present for DirectFiles/Binary/Polygons.
3. KPI4: verify ETL overhead is non-zero for Binary/Polygons and 0 for DirectFiles.

## 8) Deliverables to send back
1. Terminal logs for all phases.
2. results/unet2d_train_results.json
3. results/unet2d_kpi.json
4. Git commit hash or patch with any GPU tuning changes.
