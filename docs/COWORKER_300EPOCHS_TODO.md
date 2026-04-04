# 🚀 Phase C Training (300 epochs) — Instructions pour Coworker

## ⚠️ IMPORTANT: Dossiers à Utiliser

- **Utilise toujours `1_ETL/`** (ne pas utiliser `ETL/`)
  - Extract: `1_ETL/Extract/`
  - Transform: `1_ETL/Transform/`
  - Load: `1_ETL/Load/`

- **Utilise `2_data_augmentation/`** pour les augmentations
- **Utilise `3_Data_Partitionement/`** pour la partition

## 📋 Pré-requisites

Vérifie que tu as complété les **Phases A, B, D** en local avant de relancer Phase C:

```powershell
# Phase A: Sanity train (2 epochs, local CPU check)
python 4_Unet2D/train_unet2d_compare.py `
  --image-dir $Env:TOPBRAIN_IMAGE_DIR `
  --label-dir $Env:TOPBRAIN_LABEL_DIR `
  --partition-file 3_Data_Partitionement/partition_materialized.json `
  --epochs 2 `
  --augment

# Phase B: ETL 2D generation (préparation données MongoDB)
python run_unet2d_phases.py --phase b

# Phase D: KPI benchmarking (throughput, occupancy, overhead)
python run_unet2d_phases.py --phase d
```

## 🎯 Phase C: Training UNet2D (300 epochs)

**Configuration actuelle:**
- **Epochs:** 300 (setup dans `.env: TOPBRAIN_2D_EPOCHS=300`)
- **Data augmentation:** ACTIVE (flips, rotations, gaussian noise, gamma correction)
- **Strategies:** 3 comparées en parallèle (DirectFiles, Binary, Polygons)
- **Device:** GPU (auto-détection CUDA)
- **Batch size:** 8 (default)

### Commande Phase C:

```powershell
cd C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Project

python 4_Unet2D/train_unet2d_compare.py `
  --image-dir $Env:TOPBRAIN_IMAGE_DIR `
  --label-dir $Env:TOPBRAIN_LABEL_DIR `
  --partition-file 3_Data_Partitionement/partition_materialized.json `
  --fold fold_1 `
  --epochs 300 `
  --augment `
  --num-workers 0
```

**Attendu:**
- ~45-60 minutes de training (selon GPU)
- Affiche les metrics chaque epoch (train_loss, val_loss, dice, iou, combined_score)
- Génère JSON avec **historique complet** `results/unet2d_train_results.json`
- Sauvegarde checkpoints best: `4_Unet2D/checkpoints/unet2d_best_*.pth`

### Output Expected:

```json
{
  "fold": "fold_1",
  "strategies": [
    {
      "strategy": "directfiles",
      "best_combined": 0.2638,
      "best_epoch": 148,
      "train_slices": 1024,
      "val_slices": 256,
      "epochs": [
        {"epoch": 1, "train_loss": ..., "val_loss": ..., "dice_fg": ..., "iou_fg": ..., "combined_score": ...},
        ...
        {"epoch": 300, "train_loss": ..., "val_loss": ..., "dice_fg": ..., "iou_fg": ..., "combined_score": ...}
      ]
    },
    {...}, {...}
  ]
}
```

## 📊 Après Phase C: Visualisation

Une fois Phase C terminée, génère les courbes d'apprentissage:

```powershell
python visualize_unet2d_training.py results/unet2d_train_results.json
```

Génère 5 figures dans `results/plots/`:
1. `loss_curves_fold_1.png` — train/val loss
2. `dice_curves_fold_1.png` — dice convergence
3. `iou_curves_fold_1.png` — IoU convergence
4. `combined_curves_fold_1.png` — combined score evolution
5. `summary_stats_fold_1.png` — barplots résumés

## 🔧 Variables d'Environnement (vérifications)

Tous ces chemins doivent être valides dans `.env`:

```
TOPBRAIN_IMAGE_DIR=C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Data_Release_Batches1n2_081425\...\imagesTr_topbrain_ct
TOPBRAIN_LABEL_DIR=C:\Users\LENOVO\Desktop\PFFECerine\TopBrain_Data_Release_Batches1n2_081425\...\labelsTr_topbrain_ct
TOPBRAIN_PARTITION_FILE=3_Data_Partitionement/partition_materialized.json
TOPBRAIN_2D_EPOCHS=300
TOPBRAIN_2D_BINARY_COLLECTION=MultiClassPatients2D_Binary
TOPBRAIN_2D_POLYGON_COLLECTION=MultiClassPatients2D_Polygons
TOPBRAIN_2D_CHECKPOINT_DIR=4_Unet2D/checkpoints
TOPBRAIN_2D_TRAIN_RESULTS_JSON=results/unet2d_train_results.json
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=TopBrain_DB
```

## ✅ Checklist Phase C

- [ ] Phase A (sanity train 2 epochs) — réussi
- [ ] Phase B (ETL 2D via `1_ETL/Load/load_t6_mongodb_insert_2d.py`) — réussi
- [ ] `.env` configuré avec `TOPBRAIN_2D_EPOCHS=300`
- [ ] MongoDB collecte `MultiClassPatients2D_Binary` et `MultiClassPatients2D_Polygons` créées
- [ ] Partition `3_Data_Partitionement/partition_materialized.json` existe
- [ ] Augmentation `2_data_augmentation/` configurée
- [ ] GPU disponible et CUDA fonctionnel
- [ ] Lance Phase C (300 epochs, ~1h)
- [ ] Génère visualisation avec `visualize_unet2d_training.py`
- [ ] Push résultats et plots vers GitHub

## 🐛 Troubleshooting

### "Empty dataset for strategy=..."
```
Cause: Phase B ETL pas lancée ou données pas créées
Fix: Lance Phase B d'abord → re-lance Phase C
```

### "CUDA out of memory"
```
Cause: batch_size trop grand pour GPU
Fix: Réduis --batch-size 4 ou 2
```

### "partition_materialized.json not found"
```
Cause: Phase non complète
Fix: Lance 3_Data_Partitionement/partition_data.py d'abord
```

### "MongoDB connection refused"
```
Cause: MongoDB pas lancé
Fix: Démarre MongoDB et vérifie MONGO_URI dans .env
```

## 📝 Rappel: Dossiers à Utiliser

| Composant | Chemin CORRECT | Ne pas utiliser |
|-----------|---|---|
| ETL Extract | `1_ETL/Extract/` | `ETL/Extract/` |
| ETL Transform | `1_ETL/Transform/` | `ETL/Transform/` |
| ETL Load 2D | `1_ETL/Load/load_t6_mongodb_insert_2d.py` | `ETL/Load/` |
| Augmentation | `2_data_augmentation/` | ✓ correct |
| Partition | `3_Data_Partitionement/partition_data.py` | ✓ correct |
| UNet2D Train | `4_Unet2D/train_unet2d_compare.py` | ✓ correct |

---

**Questions?** Vérifie les logs du terminal ou contact l'équipe.

Bonne chance! 🚀
