# TopBrain Project - Segmentation Vasculaire Cérébrale 3D

Projet de segmentation automatique des vaisseaux cérébraux à partir d'angiographies CT (CTA) en 3D, utilisant un U-Net 3D avec attention gates.

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Dataset](#dataset)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Résultats](#résultats)

---

## 🎯 Aperçu

Ce projet implémente un pipeline complet pour la segmentation vasculaire cérébrale :

- **ETL Pipeline** : Extraction et stockage des données NIfTI dans MongoDB
- **Dataset PyTorch** : Chargement optimisé avec augmentation de données 3D
- **U-Net 3D** : Architecture avec attention gates pour segmentation précise
- **Training** : Entraînement avec validation, checkpoints et métriques (Dice, IoU)
- **Benchmark** : Comparaison MongoDB vs fichiers pour performances

### Classes segmentées

1. **MCA** - Middle Cerebral Artery (Artère cérébrale moyenne)
2. **ACA** - Anterior Cerebral Artery (Artère cérébrale antérieure)
3. **Vertebral** - Vertebral Artery (Artère vertébrale)
4. **Basilar** - Basilar Artery (Artère basilaire)
5. **PCA** - Posterior Cerebral Artery (Artère cérébrale postérieure)

---

## 🏗️ Architecture

### U-Net 3D avec Attention Gates

```
Encoder (contracting) → Bottleneck → Decoder (expanding)
     ↓                                    ↑
  MaxPool3D                          ConvTranspose3D
     ↓                                    ↑
Skip connections + Attention Gates ───────┘
```

**Caractéristiques** :
- Convolutions 3D avec BatchNorm et ReLU
- Attention gates pour focaliser sur les petits vaisseaux
- Combined Loss (CrossEntropy + Dice) pour gérer le déséquilibre de classes

---

## 🔧 Prérequis

- **Python** : 3.10+
- **MongoDB** : 4.0+ (pour le stockage des métadonnées)
- **GPU** : Recommandé (CUDA compatible) pour l'entraînement
- **RAM** : 16 GB minimum (32 GB recommandé pour volumes 3D)

---

## 📦 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/votre-username/TopBrain_Project.git
cd TopBrain_Project
```

### 2. Créer l'environnement virtuel

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Démarrer MongoDB

**Windows** :
```powershell
Start-Service -Name MongoDB
```

**Linux/Mac** :
```bash
sudo systemctl start mongod
```

---

## 📊 Dataset

### Format attendu

Les données doivent être au format **NIfTI** (`.nii.gz`) :

```
data/raw/TopBrain_Data_Release_Batches1n2_081425/
├── imagesTr_topbrain_ct/
│   ├── sub_001_0000.nii.gz
│   ├── sub_002_0000.nii.gz
│   └── ...
└── labelsTr_topbrain_ct/
    ├── sub_001.nii.gz
    ├── sub_002.nii.gz
    └── ...
```

### Configuration des chemins

Modifier les chemins dans [src/data/etl_pipeline.py](src/data/etl_pipeline.py#L30-L36) :

```python
class Config:
    IMAGE_DIR = r"C:\path\to\imagesTr_topbrain_ct"
    LABEL_DIR = r"C:\path\to\labelsTr_topbrain_ct"
```

---

## 🚀 Utilisation

### 1. Exécuter l'ETL Pipeline

Charger les données NIfTI dans MongoDB :

```bash
python src/data/etl_pipeline.py
```

**Sortie attendue** :
```
✓ Connexion à MongoDB établie : TopBrain_DB
✓ Collections et index initialisés
📁 25 patients à traiter
Traitement ETL: 100%|██████████| 25/25
✓ Patients traités avec succès : 25
📊 Total segments extraits : 764
```

### 2. Tester le Dataset

Vérifier que le dataset charge correctement :

```bash
python src/data/dataset_3d.py
```

### 3. Entraîner le modèle

Lancer l'entraînement U-Net 3D :

```bash
python src/train.py
```

**Paramètres d'entraînement** (dans [src/train.py](src/train.py#L25-L51)) :

```python
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
TARGET_SIZE = (128, 128, 64)  # Ajuster selon GPU
```

### 4. Benchmark MongoDB vs Fichiers

Comparer les temps d'exécution :

```bash
python src/benchmark_unet.py --patient-id 1 --runs 3
```

---

## 📁 Structure du projet

```
TopBrain_Project/
├── checkpoints/           # Modèles sauvegardés
├── logs/                  # Historiques d'entraînement
├── data/
│   ├── raw/              # Données NIfTI (non suivies par git)
│   └── processed/        # Données prétraitées
├── src/
│   ├── data/
│   │   ├── etl_pipeline.py      # Pipeline ETL MongoDB
│   │   └── dataset_3d.py        # Dataset PyTorch 3D
│   ├── models/
│   │   └── unet3d_model.py      # Architecture U-Net 3D
│   ├── utils/
│   │   └── visualization.py     # Outils de visualisation
│   ├── train.py                 # Script d'entraînement
│   └── benchmark_unet.py        # Benchmark performances
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Résultats

### Métriques de segmentation

Le modèle est évalué avec :

- **Dice Coefficient** : Mesure de similitude (0-1)
- **IoU (Intersection over Union)** : Précision de la segmentation
- **Loss combinée** : CrossEntropy + Dice Loss

### Exemple de sortie d'entraînement

```
Epoch 10/100
Train Loss: 0.3254 | Val Loss: 0.3012
Train Dice (MCA): 0.7821 | Val Dice (MCA): 0.8103
Learning Rate: 1.00e-04
✓ Best model saved: checkpoints/best_model.pth
```

---

## 🛠️ Technologies utilisées

- **PyTorch** : Deep learning framework
- **MongoDB** : Base de données NoSQL pour métadonnées
- **NiBabel** : Lecture de fichiers NIfTI
- **NumPy / SciPy** : Traitement numérique 3D
- **Matplotlib** : Visualisation

---

## 📝 Citation

Si vous utilisez ce code, veuillez citer le dataset TopBrain :

```
@dataset{topbrain2025,
  title={TopBrain: Cerebral Vascular Segmentation Dataset},
  author={...},
  year={2025}
}
```

---

## 📧 Contact

Pour toute question : [votre-email@example.com]

---

## 📜 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.