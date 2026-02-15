"""
U-Net 3D pour Segmentation Vasculaire Cérébrale
Architecture adaptée aux volumes 3D avec attention spatiale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ================== BLOCS DE CONVOLUTION 3D ==================
class Conv3DBlock(nn.Module):
    """Bloc de convolution 3D : Conv -> BatchNorm -> ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True
    ):
        super(Conv3DBlock, self).__init__()
        
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batchnorm
            )
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DoubleConv3D(nn.Module):
    """Double convolution 3D (pattern classique U-Net)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None
    ):
        super(DoubleConv3D, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            Conv3DBlock(in_channels, mid_channels),
            Conv3DBlock(mid_channels, out_channels)
        )
    
    def forward(self, x):
        return self.double_conv(x)


# ================== MODULE D'ATTENTION 3D ==================
class AttentionGate3D(nn.Module):
    """
    Attention Gate pour améliorer la segmentation des petits vaisseaux
    Focalise le réseau sur les régions d'intérêt
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Nombre de canaux du gating signal (encoder)
            F_l: Nombre de canaux du signal skip connection
            F_int: Nombre de canaux intermédiaires
        """
        super(AttentionGate3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal de l'encoder (résolution basse)
            x: Signal de la skip connection (résolution haute)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi  # Pondération par attention


# ================== U-NET 3D COMPLET ==================
class UNet3D(nn.Module):
    """
    U-Net 3D pour segmentation vasculaire cérébrale
    Architecture avec attention gates et skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,  # Fond + 5 classes vasculaires
        base_channels: int = 32,  # Réduit pour économiser mémoire GPU
        use_attention: bool = True
    ):
        """
        Args:
            in_channels: Nombre de canaux en entrée (1 pour CTA)
            out_channels: Nombre de classes de segmentation
            base_channels: Nombre de canaux de base (doublé à chaque niveau)
            use_attention: Activer les attention gates
        """
        super(UNet3D, self).__init__()
        
        self.use_attention = use_attention
        
        # Encoder (contracting path)
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)
        
        # Decoder (expanding path)
        self.upconv4 = nn.ConvTranspose3d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2
        )
        if use_attention:
            self.att4 = AttentionGate3D(base_channels * 8, base_channels * 8, base_channels * 4)
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose3d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )
        if use_attention:
            self.att3 = AttentionGate3D(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )
        if use_attention:
            self.att2 = AttentionGate3D(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )
        if use_attention:
            self.att1 = AttentionGate3D(base_channels, base_channels, base_channels // 2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)
        
        # Couche finale de classification
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 1, H, W, D]
        
        Returns:
            Output tensor [B, out_channels, H, W, D]
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder avec skip connections
        dec4 = self.upconv4(bottleneck)
        if self.use_attention:
            enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if self.use_attention:
            enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if self.use_attention:
            enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if self.use_attention:
            enc1 = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Classification finale
        output = self.final_conv(dec1)
        
        return output


# ================== FONCTIONS DE PERTE ==================
class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation multi-classe
    Meilleure gestion du déséquilibre de classes que CrossEntropy
    """
    
    def __init__(self, smooth: float = 1.0, weight: torch.Tensor = None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Prédictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D]
        """
        # Conversion en probabilités
        pred = F.softmax(pred, dim=1)
        
        # One-hot encoding du target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calcul du Dice par classe
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Moyenne pondérée par classe
        if self.weight is not None:
            dice = dice * self.weight.unsqueeze(0)
        
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combinaison de CrossEntropy et Dice Loss"""
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


# ================== TEST DU MODÈLE ==================
def test_model():
    """Test rapide du modèle"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Création du modèle
    model = UNet3D(
        in_channels=1,
        out_channels=6,
        base_channels=32,
        use_attention=True
    ).to(device)
    
    # Comptage des paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 1
    input_shape = (batch_size, 1, 128, 128, 64)  # [B, C, H, W, D]
    
    x = torch.randn(input_shape).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, 6, 128, 128, 64]")
    
    # Test de la loss
    target = torch.randint(0, 6, (batch_size, 128, 128, 64)).to(device)
    
    criterion = CombinedLoss()
    loss = criterion(output, target)
    
    print(f"\nLoss value: {loss.item():.4f}")


if __name__ == "__main__":
    test_model()