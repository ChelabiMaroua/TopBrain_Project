# TopBrain Project - Guide Complet (Debut -> Fin)

Ce document explique le workflow complet du projet:

1. ETL (creation des collections MongoDB pour 3D et 2D)
2. Data augmentation
3. Partitionnement K-Fold
4. Entrainement UNet2D (strategies: directfiles, binary, polygons) sur fold_1 -> fold_5
5. Entrainement UNet3D (strategies: directfiles, binary, polygons) sur fold_1 -> fold_5
6. Tableau recapitulatif par strategie
7. Courbes train vs val

Tout est en commandes executables depuis la racine du projet.

## 0) Prerequis

- Python environment avec: torch, numpy, nibabel, opencv-python, pymongo, python-dotenv, monai, matplotlib, pandas
- MongoDB actif
- Donnees NIfTI disponibles:
  - images: .../imagesTr_topbrain_ct
  - labels: .../labelsTr_topbrain_ct

Exemple installation rapide:

```powershell
pip install torch numpy nibabel opencv-python pymongo python-dotenv monai matplotlib pandas
```

Depuis la racine du projet:

```powershell
cd C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Project
```

## 1) ETL - Creation des collections MongoDB

### 1.1 Collection 3D binaire/multiclasse (utilisee par UNet3D strategy=binary)

Important: pour une segmentation multiclasse (0..5), garder les labels multiclasses avec --keep-multiclass-labels.

```powershell
python 1_ETL/Load/load_t5_mongodb_insert.py `
  --image-dir "C:/path/to/imagesTr_topbrain_ct" `
  --label-dir "C:/path/to/labelsTr_topbrain_ct" `
  --target-size 128 128 64 `
  --db-name TopBrain_DB `
  --collection MultiClassPatients `
  --keep-multiclass-labels
```

### 1.2 Collections 2D (utilisees par UNet2D strategy=binary et strategy=polygons)

```powershell
python 1_ETL/Load/load_t6_mongodb_insert_2d.py `
  --image-dir "C:/path/to/imagesTr_topbrain_ct" `
  --label-dir "C:/path/to/labelsTr_topbrain_ct" `
  --target-size 128 128 64 `
  --db-name TopBrain_DB `
  --binary-collection MultiClassPatients2D_Binary `
  --polygon-collection MultiClassPatients2D_Polygons
```

### 1.3 Collection 3D polygons (requise pour UNet3D strategy=polygons)

Le trainer 3D lit par defaut la collection PolygonPatients, mais le script de creation 3D polygon n'est pas dans ce repo.

Si cette collection existe deja, vous pouvez lancer --strategy all en 3D.
Sinon, utilisez temporairement --strategy directfiles ou --strategy binary pour la partie 3D.

## 2) Data augmentation

- UNet2D applique l'augmentation online avec --augment (active par defaut).
- UNet3D compare n'applique pas de flag --augment dans ce script.
- Visualisation MONAI possible avec:

```powershell
python 2_data_augmentation/visualize_patient_monai.py `
  --patient-id 001 `
  --target-size 128 128 64 `
  --mongo-uri "mongodb://localhost:27017" `
  --db-name TopBrain_DB `
  --collection MultiClassPatients
```

## 3) Partitionnement K-Fold

Genere le fichier partition_materialized.json depuis MongoDB.

```powershell
python 3_Data_Partitionement/partition_data.py `
  --mongo-uri "mongodb://localhost:27017" `
  --db-name TopBrain_DB `
  --collection MultiClassPatients `
  --k 5 `
  --test-ratio 0.2 `
  --seed 42 `
  --output 3_Data_Partitionement/partition_materialized.json
```

## 4) Entrainement UNet2D - all strategies - fold_1 a fold_5

Commande PowerShell (boucle 5 folds):

```powershell
New-Item -ItemType Directory -Force -Path results/kfold | Out-Null
$folds = @("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")

foreach ($fold in $folds) {
  python 4_Unet2D/train_unet2d_compare.py `
    --strategy all `
    --image-dir "C:/path/to/imagesTr_topbrain_ct" `
    --label-dir "C:/path/to/labelsTr_topbrain_ct" `
    --partition-file 3_Data_Partitionement/partition_materialized.json `
    --fold $fold `
    --epochs 150 `
    --batch-size 4 `
    --lr 5e-5 `
    --base-channels 64 `
    --sampling-mode class-aware `
    --foreground-boost 6.0 `
    --class-boosts "1:1.0,2:1.5,3:3.0,4:8.0,5:5.0" `
    --early-stopping 40 `
    --min-epochs 60 `
    --num-workers 2 `
    --save-dir 4_Unet2D/checkpoints `
    --output-json "results/kfold/unet2d_train_results_$fold.json"
}
```

## 5) Entrainement UNet3D - all strategies - fold_1 a fold_5

Important: pour 3D, batch-size=1 et base-channels=32 sont recommandes pour eviter OOM/timeout.

```powershell
New-Item -ItemType Directory -Force -Path results/kfold | Out-Null
$folds = @("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")

foreach ($fold in $folds) {
  python 4_Unet3D/train_unet3d_compare.py `
    --strategy all `
    --image-dir "C:/path/to/imagesTr_topbrain_ct" `
    --label-dir "C:/path/to/labelsTr_topbrain_ct" `
    --partition-file 3_Data_Partitionement/partition_materialized.json `
    --fold $fold `
    --epochs 150 `
    --batch-size 1 `
    --lr 5e-5 `
    --base-channels 32 `
    --sampling-mode class-aware `
    --foreground-boost 6.0 `
    --class-boosts "1:1.0,2:1.5,3:3.0,4:8.0,5:5.0" `
    --early-stopping 40 `
    --min-epochs 60 `
    --num-workers 2 `
    --save-dir 4_Unet3D/checkpoints `
    --output-json "results/kfold/unet3d_train_results_$fold.json"
}
```

Si la collection 3D polygons n'existe pas encore, remplacez --strategy all par --strategy directfiles ou --strategy binary.

## 6) Tableau recapitulatif par strategie (2D et 3D)

Le bloc ci-dessous genere:

- results/kfold/summary_unet2d.csv
- results/kfold/summary_unet3d.csv
- results/kfold/summary_unet2d.md
- results/kfold/summary_unet3d.md

```powershell
@'
import glob
import json
import os
import pandas as pd

def summarize(pattern, out_csv, out_md):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fold = payload.get("fold", os.path.basename(path).replace(".json", ""))
        for s in payload.get("strategies", []):
            rows.append({
                "fold": fold,
                "strategy": s.get("strategy"),
                "best_combined": s.get("best_combined", 0.0),
                "best_epoch": s.get("best_epoch", 0),
                "epochs_ran": len(s.get("epochs", [])),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No data for pattern: {pattern}")
        return

    df.to_csv(out_csv, index=False)

    agg = (
        df.groupby("strategy", as_index=False)
          .agg(
              mean_best_combined=("best_combined", "mean"),
              std_best_combined=("best_combined", "std"),
              mean_best_epoch=("best_epoch", "mean"),
          )
          .sort_values("mean_best_combined", ascending=False)
    )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Strategy recap\n\n")
        f.write("## Per fold\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Mean by strategy\n\n")
        f.write(agg.to_markdown(index=False, floatfmt=".4f"))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")

summarize("results/kfold/unet2d_train_results_fold_*.json", "results/kfold/summary_unet2d.csv", "results/kfold/summary_unet2d.md")
summarize("results/kfold/unet3d_train_results_fold_*.json", "results/kfold/summary_unet3d.csv", "results/kfold/summary_unet3d.md")
'@ | python
```

## 7) Courbes train vs val (loss + combined) pour chaque fold/strategie

Le bloc ci-dessous genere des PNG dans results/plots:

- unet2d_fold_X_strategy_Y_curves.png
- unet3d_fold_X_strategy_Y_curves.png

```powershell
@'
import glob
import json
import os
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

def render_curves(pattern, prefix):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for {pattern}")
        return

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fold = payload.get("fold", os.path.basename(path).replace(".json", ""))

        for s in payload.get("strategies", []):
            strategy = s.get("strategy", "unknown")
            epochs = s.get("epochs", [])
            if not epochs:
                continue

            x = [e.get("epoch", i + 1) for i, e in enumerate(epochs)]
            train_loss = [e.get("train_loss", 0.0) for e in epochs]
            val_loss = [e.get("val_loss", 0.0) for e in epochs]
            val_combined = [e.get("combined_score", 0.0) for e in epochs]
            val_dice = [e.get("dice_fg", 0.0) for e in epochs]
            val_iou = [e.get("iou_fg", 0.0) for e in epochs]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            axes[0].plot(x, train_loss, label="train_loss")
            axes[0].plot(x, val_loss, label="val_loss")
            axes[0].set_title(f"{prefix} | {fold} | {strategy} | Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(alpha=0.3)
            axes[0].legend()

            axes[1].plot(x, val_dice, label="val_dice_fg")
            axes[1].plot(x, val_iou, label="val_iou_fg")
            axes[1].plot(x, val_combined, label="val_combined")
            axes[1].set_title(f"{prefix} | {fold} | {strategy} | Val metrics")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].grid(alpha=0.3)
            axes[1].legend()

            out = f"results/plots/{prefix.lower()}_{fold}_{strategy}_curves.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved: {out}")

render_curves("results/kfold/unet2d_train_results_fold_*.json", "UNet2D")
render_curves("results/kfold/unet3d_train_results_fold_*.json", "UNet3D")
'@ | python
```

## 8) Outputs finaux attendus

- results/kfold/unet2d_train_results_fold_1.json ... fold_5.json
- results/kfold/unet3d_train_results_fold_1.json ... fold_5.json
- results/kfold/summary_unet2d.md
- results/kfold/summary_unet3d.md
- results/plots/unet2d_fold_..._curves.png
- results/plots/unet3d_fold_..._curves.png

## 9) Conseils anti-timeout / anti-OOM

- UNet3D: batch-size 1, base-channels 32
- Utiliser early stopping (ex: --early-stopping 40 --min-epochs 60)
- Reduire num-workers a 0 ou 1 si instable
- Sur laptop Windows branche sur secteur et desactive la veille:

```powershell
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
```
