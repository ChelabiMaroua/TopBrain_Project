"""
stage3_training_improvements.py
================================
Deux améliorations pour train_level2.py (stage-3, 41 classes) :

  1. compute_stage3_weights()
     Calcule des poids CE calibrés par taille de classe (Tier A/B/C).
     Drop-in replacement pour --auto-class-weights, avec contrôle manuel
     des Tier C (Pcom, Acom, AICA) pour éviter de polluer le gradient.

  2. FamilyPriorLoss
     Wraps DiceCELoss et ajoute un terme de régularisation spatiale :
     dans les zones où family_map prédit la famille F, on pénalise fort
     les logits des classes qui n'appartiennent PAS à F.
     → Le modèle apprend à « chercher au bon endroit ».

Usage dans train_level2.py :
-----------------------------
from stage3_training_improvements import compute_stage3_weights, FamilyPriorLoss

# 1. Remplacer --auto-class-weights par les poids calibrés :
weights = compute_stage3_weights(train_counts, num_classes=41, device=device)

# 2. Remplacer DiceCELossWrapper par FamilyPriorLoss :
criterion = FamilyPriorLoss(
    lambda_dice=2.0,
    lambda_ce=0.5,
    lambda_prior=0.3,      # poids du terme prior (0 = désactivé = comportement original)
    ce_weight=weights,
    num_classes=41,
    num_families=8,        # 0=BG + G1..G7 (avec G7 veineux)
)

# Dans le forward du training loop — inchangé, x[1] est déjà la family_map :
#   x : [B, 2, H, W, D]  (x[:,0]=CTA, x[:,1]=family_map normalisée)
#   y : [B, H, W, D]     (labels 0-40)
loss = criterion(logits, y, x[:, 1:2, ...])   # passer family_map en 3e arg
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


# =============================================================================
# 1.  Mapping famille → classes fines  (à mettre à jour si tu passes à G7)
# =============================================================================

# Avec ton nouveau mapping G1..G7 + G0=fond
# Clé = id famille (1-7), valeur = liste des class IDs fine (1-40) appartenant à cette famille
FAMILY_TO_CLASSES: Dict[int, List[int]] = {
    1: [1, 23, 24, 25, 26, 27, 28, 29, 30],          # G1 Vertébro-Basilaire
    2: [4, 6, 31, 32, 33, 34],                        # G2 Carotides + branches
    3: [5, 7, 17, 18, 19, 20],                        # G3 ACM
    4: [11, 12, 13, 14, 15, 16],                      # G4 ACA
    5: [2, 3, 21, 22],                                # G5 ACP
    6: [8, 9, 10],                                    # G6 Communicantes
    7: [35, 36, 37, 38, 39, 40],                      # G7 Veineux
}

# Tiers de taille (basés sur class_hierarchy.json + explore_level2_dataset)
# Tier A ≥ 300 vox · Tier B 100-300 vox · Tier C < 100 vox (sub-voxel)
TIER_A: List[int] = [1, 2, 3, 4, 5, 6, 7, 11, 12, 17, 19, 23, 24, 35, 36, 37, 38, 39, 40]
TIER_B: List[int] = [13, 14, 15, 16, 18, 20, 21, 22, 25, 26, 29, 30, 31, 32, 33, 34]
TIER_C: List[int] = [8, 9, 10, 27, 28]   # Pcom L/R, Acom, AICA L/R

# Poids manuels par tier (surcharge l'auto si explicitement passés)
TIER_WEIGHTS: Dict[str, float] = {
    "A": 1.0,   # grandes classes → poids nominal
    "B": 2.5,   # classes incertaines → boostées
    "C": 0.15,  # sub-voxel → gardées dans le modèle mais quasi-muettes
}


# =============================================================================
# 2.  compute_stage3_weights
# =============================================================================

def compute_stage3_weights(
    train_counts: np.ndarray,
    num_classes: int = 41,
    device: Optional[torch.device] = None,
    bg_weight: float = 0.05,
    cap: float = 20.0,
    tier_override: bool = True,
) -> torch.Tensor:
    """
    Calcule les poids CE pour le stage-3.

    Stratégie :
      - Base  : median-frequency weighting (comme --auto-class-weights)
      - Patch : les classes Tier C sont ramenées à TIER_WEIGHTS['C']
                pour ne pas polluer le gradient avec leur Dice instable.

    Args:
        train_counts : np.ndarray [num_classes] — nb de voxels par classe dans le train set.
                       Obtenu via args.log_label_distribution ou pré-calculé offline.
        num_classes  : 41 par défaut.
        device       : torch device cible.
        bg_weight    : poids du fond (classe 0). Défaut 0.05.
        cap          : plafond multiplicatif pour les classes rares. Défaut 20.
        tier_override: si True, applique les poids Tier C en surcharge de l'auto.

    Returns:
        torch.Tensor [num_classes] float32, normalisé (mean ≈ 1.0 sur FG).
    """
    counts = np.array(train_counts, dtype=np.float64)
    if len(counts) != num_classes:
        raise ValueError(f"train_counts length {len(counts)} ≠ num_classes {num_classes}")

    eps = 1e-6
    freqs = counts / max(counts.sum(), 1.0)

    fg_freqs = freqs[1:]                              # classes 1..40
    safe_fg  = np.maximum(fg_freqs, eps)
    median_fg = float(np.median(safe_fg))

    weights = np.ones(num_classes, dtype=np.float32)
    weights[0] = bg_weight
    weights[1:] = np.clip((median_fg / safe_fg).astype(np.float32), 0.0, cap)

    if tier_override:
        # Tier C : sub-voxel → poids très faibles
        for c in TIER_C:
            if c < num_classes:
                weights[c] = TIER_WEIGHTS["C"]
        # Tier B : classes incertaines → légèrement boostées si auto les a sous-estimées
        for c in TIER_B:
            if c < num_classes:
                weights[c] = max(weights[c], TIER_WEIGHTS["B"])
        # Tier A : on laisse l'auto mais on plafonne à 5 pour ne pas écraser les petits tiers
        for c in TIER_A:
            if c < num_classes:
                weights[c] = min(weights[c], 5.0)

    # Normaliser sur le FG uniquement (la CE loss de MONAI normalise déjà,
    # mais être explicite évite les surprises de magnitude)
    fg_mean = float(weights[1:].mean())
    if fg_mean > 0:
        weights[1:] /= fg_mean

    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)

    print("[stage3_weights] Tier breakdown :")
    print(f"  Tier A ({len(TIER_A)} classes) → mean={weights[TIER_A].mean():.3f}  "
          f"min={weights[TIER_A].min():.3f}  max={weights[TIER_A].max():.3f}")
    print(f"  Tier B ({len(TIER_B)} classes) → mean={weights[TIER_B].mean():.3f}  "
          f"min={weights[TIER_B].min():.3f}  max={weights[TIER_B].max():.3f}")
    print(f"  Tier C ({len(TIER_C)} classes) → mean={weights[TIER_C].mean():.3f}  "
          f"(sub-voxel, poids volontairement faibles)")
    print(f"  BG     weight = {weights[0]:.3f}")

    return tensor


# =============================================================================
# 3.  Matrice prior  family → classes autorisées
# =============================================================================

def build_family_prior_mask(
    num_classes: int = 41,
    num_families: int = 8,
) -> torch.Tensor:
    """
    Retourne un masque booléen [num_families, num_classes] :
      prior_mask[f, c] = True  si la classe c appartient à la famille f
                         False sinon (classe interdite dans cette famille)

    Famille 0 = fond → seule la classe 0 est autorisée.
    Si une classe n'est dans aucune famille (ne devrait pas arriver), elle reste True partout.
    """
    # Par défaut tout est interdit dans les familles non-fond
    mask = torch.zeros(num_families, num_classes, dtype=torch.bool)

    # Famille 0 = fond
    mask[0, 0] = True

    for fam_id, class_ids in FAMILY_TO_CLASSES.items():
        if fam_id >= num_families:
            continue
        # La classe fond reste toujours "autorisée" (le modèle peut prédire fond partout)
        mask[fam_id, 0] = True
        for c in class_ids:
            if c < num_classes:
                mask[fam_id, c] = True

    return mask   # [num_families, num_classes]


# =============================================================================
# 4.  FamilyPriorLoss
# =============================================================================

class FamilyPriorLoss(nn.Module):
    """
    Loss combinée pour le stage-3 :

        L = L_DiceCE(logits, y)  +  lambda_prior * L_prior(logits, family_map)

    L_prior pénalise les logits des classes *incompatibles* avec la famille prédite
    par le stage-2. Concrètement, pour chaque voxel v dont la famille prédite est f,
    on veut que softmax(logits)[v, classes_interdites_de_f] → 0.

    Implémentation : BCE sur les probabilités des classes interdites
    (target = 0 dans ces zones), pondérée par la confiance famille.

    Args:
        lambda_dice  : poids du terme Dice (inchangé vs DiceCELossWrapper).
        lambda_ce    : poids du terme CE   (inchangé).
        lambda_prior : poids du terme prior spatial. 0 = désactivé.
        ce_weight    : tensor [num_classes] — poids CE par classe.
        num_classes  : 41.
        num_families : 8 (0=BG + G1..G7).
        prior_warmup_epochs : nombre d'epochs avant d'activer le prior
                              (laisser le modèle converger d'abord).

    Forward :
        logits      : [B, C, H, W, D]
        target      : [B, H, W, D]   labels 0-40
        family_map  : [B, 1, H, W, D] normalisée [0..1], argmax * (num_families-1)
                      → multiplier par (num_families-1) pour retrouver l'index famille.
        epoch       : int (pour le warmup)
    """

    def __init__(
        self,
        lambda_dice: float = 2.0,
        lambda_ce: float = 0.5,
        lambda_prior: float = 0.3,
        ce_weight: Optional[torch.Tensor] = None,
        num_classes: int = 41,
        num_families: int = 8,
        prior_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.num_classes = num_classes
        self.num_families = num_families
        self.prior_warmup_epochs = prior_warmup_epochs

        # Normaliser le poids CE comme dans DiceCELossWrapper
        if ce_weight is not None:
            ce_weight = ce_weight / ce_weight.mean()

        self.dice_ce = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            weight=ce_weight,
        )

        # prior_mask : [num_families, num_classes] bool — classes autorisées par famille
        prior_mask = build_family_prior_mask(num_classes, num_families)
        # forbidden_mask : [num_families, num_classes] float — 1 = classe interdite
        forbidden_mask = (~prior_mask).float()
        # Enregistrer comme buffer (pas de gradient, suit le device automatiquement)
        self.register_buffer("forbidden_mask", forbidden_mask)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        family_map: Optional[torch.Tensor] = None,
        epoch: int = 9999,
    ) -> torch.Tensor:
        """
        Args:
            logits      : [B, C, H, W, D]
            target      : [B, H, W, D]  ou  [B, 1, H, W, D]
            family_map  : [B, 1, H, W, D] normalisée [0..1]  (x[:,1:2,...] du batch)
            epoch       : epoch courante (pour warmup)
        """
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)           # → [B, 1, H, W, D]

        # ── Terme DiceCE principal (inchangé) ─────────────────────────────
        loss_main = self.dice_ce(logits, target)

        # ── Terme prior spatial ───────────────────────────────────────────
        if (
            self.lambda_prior <= 0.0
            or family_map is None
            or epoch < self.prior_warmup_epochs
        ):
            return loss_main

        loss_prior = self._compute_prior_loss(logits, family_map)
        return loss_main + self.lambda_prior * loss_prior

    def _compute_prior_loss(
        self,
        logits: torch.Tensor,       # [B, C, H, W, D]
        family_map: torch.Tensor,   # [B, 1, H, W, D] normalisée [0..1]
    ) -> torch.Tensor:
        """
        Pour chaque voxel, récupère l'index de la famille prédite et pénalise
        les logits des classes incompatibles avec cette famille.
        """
        B, C, H, W, D = logits.shape

        # Récupérer l'index famille discret pour chaque voxel
        # family_map est normalisée ÷ (num_families-1), donc on remultiplie
        fam_idx = (family_map.squeeze(1) * (self.num_families - 1))  # [B, H, W, D]
        fam_idx = fam_idx.long().clamp(0, self.num_families - 1)     # [B, H, W, D]

        # Pour chaque voxel, lookup du vecteur forbidden [C]
        # forbidden_mask : [num_families, C]  →  lookup → [B, H, W, D, C]
        forbidden_per_voxel = self.forbidden_mask[fam_idx]           # [B, H, W, D, C]
        # Transposer en [B, C, H, W, D] pour aligner avec les logits
        forbidden_per_voxel = forbidden_per_voxel.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]

        # Probabilités softmax
        probs = F.softmax(logits, dim=1)                              # [B, C, H, W, D]

        # Pénaliser : on veut probs des classes interdites → 0
        # BCE(probs_interdites, target=0)  =  -log(1 - probs_interdites)
        # On utilise une version stable : mean sur les voxels et classes interdites
        forbidden_probs = probs * forbidden_per_voxel                 # [B, C, H, W, D]

        # Éviter log(0)
        eps = 1e-6
        loss_prior = -torch.log(1.0 - forbidden_probs.clamp(0.0, 1.0 - eps))

        # Moyenner uniquement sur les voxels où une famille non-fond est prédite
        # (le fond a peu de classes interdites par définition)
        fg_mask = (fam_idx > 0).unsqueeze(1).float()                  # [B, 1, H, W, D]
        n_fg = fg_mask.sum().clamp(min=1.0)
        loss_prior = (loss_prior * fg_mask).sum() / n_fg

        return loss_prior


# =============================================================================
# 5.  Utilitaire CLI : calcul des poids offline depuis un JSON de stats
# =============================================================================

def compute_weights_from_stats_json(stats_json_path: str, output: bool = True) -> np.ndarray:
    """
    Calcule et affiche les poids depuis le JSON produit par explore_level2_dataset.py.

    Usage :
        python stage3_training_improvements.py --stats results/level2_stats.json
    """
    import json
    with open(stats_json_path) as f:
        stats = json.load(f)

    counts = np.zeros(41, dtype=np.float64)
    for key, val in stats.items():
        try:
            c = int(key)
            if 0 <= c <= 40:
                counts[c] = float(val.get("total_voxels", val.get("mean_voxels", 0)))
        except (ValueError, AttributeError):
            continue

    weights = compute_stage3_weights(counts, num_classes=41)
    w_list = weights.numpy().tolist()

    if output:
        formatted = ",".join(f"{w:.4f}" for w in w_list)
        print("\n[output] Poids prêts à coller dans --class-weights :")
        print(formatted)
        print("\n[output] Pour train_level2.py :")
        print(f"  --class-weights \"{formatted}\"")

    return weights.numpy()


# =============================================================================
# 6.  Test rapide (sans GPU, sans MongoDB)
# =============================================================================

def _smoke_test() -> None:
    print("=" * 60)
    print("SMOKE TEST — FamilyPriorLoss")
    print("=" * 60)

    B, C, H, W, D = 1, 41, 32, 32, 16

    # Poids factices (en pratique : compute_stage3_weights sur train_counts)
    fake_counts = np.random.randint(100, 50000, size=41).astype(np.float64)
    fake_counts[0] = 5_000_000   # fond très présent
    for c in TIER_C:
        fake_counts[c] = 50      # classes sub-voxel rares

    weights = compute_stage3_weights(fake_counts, num_classes=C, device=torch.device("cpu"))

    criterion = FamilyPriorLoss(
        lambda_dice=2.0,
        lambda_ce=0.5,
        lambda_prior=0.3,
        ce_weight=weights,
        num_classes=C,
        num_families=8,
        prior_warmup_epochs=0,   # activer immédiatement pour le test
    )

    logits     = torch.randn(B, C, H, W, D)
    target     = torch.randint(0, C, (B, H, W, D))
    family_map = torch.rand(B, 1, H, W, D)   # normalisée [0..1]

    # Test sans prior (epoch < warmup)
    loss_no_prior = criterion(logits, target, family_map=None)
    # Test avec prior
    loss_with_prior = criterion(logits, target, family_map=family_map, epoch=50)

    print(f"\nLoss sans prior  : {loss_no_prior.item():.4f}")
    print(f"Loss avec prior  : {loss_with_prior.item():.4f}")
    print(f"Δ (prior term)   : {(loss_with_prior - loss_no_prior).item():.4f}")

    # Vérifier le masque
    mask = build_family_prior_mask(41, 8)
    print(f"\nPrior mask shape : {mask.shape}")
    for f, classes in FAMILY_TO_CLASSES.items():
        n_allowed = mask[f].sum().item()
        print(f"  Famille G{f} : {n_allowed} classes autorisées ({classes})")

    print("\n✓ Smoke test passé")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage-3 training improvements")
    parser.add_argument("--stats", default="", help="Chemin vers level2_stats.json")
    parser.add_argument("--smoke-test", action="store_true", help="Lancer le smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    elif args.stats:
        compute_weights_from_stats_json(args.stats)
    else:
        print("Usage :")
        print("  python stage3_training_improvements.py --smoke-test")
        print("  python stage3_training_improvements.py --stats results/level2_stats.json")
