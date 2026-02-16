"""
Script d'entraînement pour U-Net 3D
Entraînement complet avec validation, checkpoints et métriques
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

# Import des modules personnalisés (à adapter)
from data.dataset_3d import TopBrainDataset3D, create_dataloaders
from models.unet3d_model import UNet3D, CombinedLoss

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION D'ENTRAÎNEMENT ==================
class TrainingConfig:
    # Chemins
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    
    # Hyperparamètres
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Architecture
    IN_CHANNELS = 1
    OUT_CHANNELS = 6
    BASE_CHANNELS = 32
    USE_ATTENTION = True
    
    # Données
    TARGET_SIZE = (128, 128, 64)
    TRAIN_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Optimisation
    EARLY_STOPPING_PATIENCE = 15
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    
    # Checkpoints
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 10
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Vérification GPU
        if not torch.cuda.is_available():
            logger.warning("⚠️  GPU non disponible - entraînement sur CPU sera très lent!")
            logger.warning(f"   PyTorch version: {torch.__version__}")
            logger.warning("   Pour GPU: Utilisez Python 3.12/3.13 avec PyTorch CUDA")


# ================== MÉTRIQUES DE SEGMENTATION ==================
class SegmentationMetrics:
    """Calcul des métriques de segmentation"""
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[int, float]:
        """
        Calcule le coefficient de Dice par classe
        
        Args:
            pred: Prédictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D]
            num_classes: Nombre de classes
        
        Returns:
            Dict avec Dice par classe
        """
        pred = torch.argmax(pred, dim=1)  # [B, H, W, D]
        
        dice_scores = {}
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores[c] = dice.item()
            else:
                dice_scores[c] = 0.0
        
        return dice_scores
    
    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[int, float]:
        """Calcule l'IoU (Intersection over Union) par classe"""
        pred = torch.argmax(pred, dim=1)
        
        iou_scores = {}
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            if union > 0:
                iou = intersection / union
                iou_scores[c] = iou.item()
            else:
                iou_scores[c] = 0.0
        
        return iou_scores


# ================== TRAINER ==================
class Trainer:
    """Classe principale pour l'entraînement du modèle"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialisation du modèle
        self.model = UNet3D(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            use_attention=config.USE_ATTENTION
        ).to(self.device)
        
        # Fonction de perte
        self.criterion = CombinedLoss()
        
        # Optimiseur
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.LR_SCHEDULER_PATIENCE,
            factor=config.LR_SCHEDULER_FACTOR,
            verbose=True
        )
        
        # Métriques
        self.metrics = SegmentationMetrics()
        
        # Historique d'entraînement
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        
        epoch_loss = 0.0
        all_dice_scores = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Déplacement sur GPU
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calcul de la perte
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            epoch_loss += loss.item()
            dice_scores = self.metrics.dice_coefficient(
                outputs.detach(),
                labels,
                self.config.OUT_CHANNELS
            )
            all_dice_scores.append(dice_scores)
            
            # Mise à jour barre de progression
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{np.mean([d[1] for d in all_dice_scores]):.4f}'  # Dice moyen (classe 1)
            })
        
        # Calcul des moyennes
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = self._compute_average_dice(all_dice_scores)
        
        return avg_loss, avg_dice
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Évalue le modèle sur le set de validation"""
        self.model.eval()
        
        epoch_loss = 0.0
        all_dice_scores = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                epoch_loss += loss.item()
                dice_scores = self.metrics.dice_coefficient(
                    outputs,
                    labels,
                    self.config.OUT_CHANNELS
                )
                all_dice_scores.append(dice_scores)
        
        avg_loss = epoch_loss / len(val_loader)
        avg_dice = self._compute_average_dice(all_dice_scores)
        
        return avg_loss, avg_dice
    
    def _compute_average_dice(self, dice_list: list) -> Dict:
        """Calcule les Dice moyens par classe"""
        avg_dice = {}
        num_classes = len(dice_list[0])
        
        for c in range(num_classes):
            scores = [d[c] for d in dice_list]
            avg_dice[c] = np.mean(scores)
        
        return avg_dice
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Checkpoint régulier
        if not self.config.SAVE_BEST_ONLY or epoch % self.config.SAVE_EVERY_N_EPOCHS == 0:
            path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")
        
        # Meilleur modèle
        if is_best:
            path = self.config.CHECKPOINT_DIR / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"✓ Best model saved: {path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Boucle d'entraînement complète"""
        logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
            logger.info(f"{'='*50}")
            
            # Entraînement
            train_loss, train_dice = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_dice = self.validate(val_loader)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Mise à jour historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rates'].append(current_lr)
            
            # Affichage des résultats
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"Train Dice (MCA): {train_dice.get(1, 0):.4f} | Val Dice (MCA): {val_dice.get(1, 0):.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Sauvegarde du meilleur modèle
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Sauvegarde de l'historique
        history_path = self.config.LOG_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*50}")


# ================== SCRIPT PRINCIPAL ==================
def main():
    # Configuration
    config = TrainingConfig()
    
    # Création des DataLoaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_split=config.TRAIN_SPLIT
    )
    
    # Initialisation du trainer
    trainer = Trainer(config)
    
    # Entraînement
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()