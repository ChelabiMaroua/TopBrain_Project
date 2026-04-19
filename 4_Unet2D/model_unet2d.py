import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class DropBlock2D(nn.Module):
    """
    DropBlock: A Regularization Method for Convolutional Networks (Ghiasi et al., 2018).

    Drops contiguous spatial blocks of activations instead of individual units.
    Much more effective than standard Dropout for dense prediction tasks like
    segmentation, because it forces the network to learn spatially distributed
    representations rather than relying on local texture patterns.

    During inference (eval mode) this is a no-op — identical to standard Dropout.

    Args:
        block_size: Side length of the square block to drop (e.g. 7).
                    Larger = more aggressive regularization.
        drop_prob:  Probability of dropping a block. Scales internally to
                    account for block_size so that the expected fraction of
                    dropped units matches drop_prob.
    """

    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = int(block_size)
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        _, _, h, w = x.shape
        bs = self.block_size

        # Compute seed drop probability so that the expected *masked* fraction
        # of the feature map equals drop_prob (accounting for block overlap).
        gamma = (
            self.drop_prob
            / (bs ** 2)
            * (h * w)
            / max(1, (h - bs + 1) * (w - bs + 1))
        )
        gamma = float(min(gamma, 1.0))

        # Sample a seed mask at full spatial resolution (one seed per channel per pixel).
        seed_mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], h, w),
                                               gamma, device=x.device, dtype=x.dtype))

        # Expand each seed into a block by max-pooling.
        block_mask = F.max_pool2d(
            seed_mask,
            kernel_size=(bs, bs),
            stride=1,
            padding=bs // 2,
        )
        # Crop to original size in case padding added an extra row/col.
        block_mask = block_mask[:, :, :h, :w]

        # Invert: 1 = keep, 0 = drop.
        block_mask = 1.0 - block_mask

        # Normalize to preserve expected activation magnitude (like inverted dropout).
        numel = block_mask.numel()
        kept = block_mask.sum()
        scale = numel / kept.clamp(min=1.0)

        return x * block_mask * scale

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}, drop_prob={self.drop_prob}"


class DoubleConv2D(nn.Module):
    """
    Two consecutive Conv-GN-ReLU blocks with optional DropBlock after the second activation.

    The DropBlock is placed *after* the second ReLU so it regularizes the final
    feature map produced by this block before it is passed to the next stage
    (pooling or upsampling) or used as a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropblock: DropBlock2D | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # DropBlock applied after the block's final activation (or None = disabled).
        self.dropblock: DropBlock2D | None = dropblock

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        if self.dropblock is not None:
            x = self.dropblock(x)
        return x


class UNet2D(nn.Module):
    """
    2-D UNet with DropBlock regularization on the encoder and decoder paths.

    DropBlock parameters:
        dropblock_prob  – drop probability per block (0 = disabled).
                          Typical range: 0.05–0.15.  Start with 0.10.
        dropblock_size  – spatial block side length.
                          7 works well for 256×256 inputs;
                          use 5 for 128×128 or 96×96 patches.

    The bottleneck intentionally uses a slightly higher drop rate
    (dropblock_prob * 1.5, capped at 0.3) because it operates at the
    lowest spatial resolution and is most prone to over-fitting texture.

    Skip connections share the DropBlock instance of their encoder stage,
    meaning they are regularized *before* being concatenated — consistent
    with the original DropBlock paper's recommendation for dense prediction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 32,
        dropblock_prob: float = 0.10,
        dropblock_size: int = 7,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        p = float(dropblock_prob)
        bs = int(dropblock_size)

        # Helper: build a DropBlock only if drop_prob > 0.
        def _db(prob: float) -> DropBlock2D | None:
            return DropBlock2D(block_size=bs, drop_prob=prob) if prob > 0.0 else None

        # --- Encoder ---
        # Shallow stages get a lighter regularization; deeper stages get full p.
        self.enc1 = DoubleConv2D(in_channels, c1, dropblock=_db(p * 0.5))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv2D(c1, c2, dropblock=_db(p * 0.75))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv2D(c2, c3, dropblock=_db(p))
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck (highest regularization) ---
        bottleneck_p = min(0.30, p * 1.5)
        self.bottleneck = DoubleConv2D(c3, c4, dropblock=_db(bottleneck_p))

        # --- Decoder (mirror of encoder) ---
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(c3 + c3, c3, dropblock=_db(p))

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(c2 + c2, c2, dropblock=_db(p * 0.75))

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        # Last decoder stage: very light regularization to preserve fine details.
        self.dec1 = DoubleConv2D(c1 + c1, c1, dropblock=_db(p * 0.25))

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)           # DropBlock(p*0.50) applied inside
        e2 = self.enc2(self.pool1(e1))   # DropBlock(p*0.75)
        e3 = self.enc3(self.pool2(e2))   # DropBlock(p*1.00)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # DropBlock(p*1.50)

        # Decoder — skip connections use already-regularized encoder features
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)