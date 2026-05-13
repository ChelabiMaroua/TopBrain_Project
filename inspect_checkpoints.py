#!/usr/bin/env python3
"""Detailed checkpoint inspection."""
import torch
from pathlib import Path

checkpoints = [
    ('stage1_binary_96', '4_Unet3D/checkpoints/stage1_binary_96/swinunetr_best_fold_1.pth'),
    ('stage2_level1_96', '5_HierarchicalSeg/checkpoints/stage2_level1_96/swinunetr_level1_best_fold_1.pth'),
    ('stage3_fine_96', '5_HierarchicalSeg/checkpoints/stage3_fine_96/swinunetr_level2_best_fold_1.pth'),
    ('stage3_level2_v2', '5_HierarchicalSeg/checkpoints/stage3_level2_v2/swinunetr_level2_best_fold_1.pth'),
]

print("\n=== CHECKPOINT STRUCTURE & METADATA ===\n")

for name, path in checkpoints:
    p = Path(path)
    if not p.exists():
        print(f"❌ {name}: NOT FOUND")
        continue
    
    try:
        size_mb = p.stat().st_size / (1024**2)
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        
        print(f"✅ {name} ({size_mb:.1f} MB)")
        
        # Check structure
        if isinstance(ckpt, dict):
            print(f"   Keys: {list(ckpt.keys())}")
            
            # Check metadata
            if 'metadata' in ckpt:
                meta = ckpt['metadata']
                print(f"   Metadata:")
                for k, v in meta.items():
                    print(f"     - {k}: {v}")
            
            # Check model state
            if 'model_state' in ckpt:
                state = ckpt['model_state']
                num_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
                num_layers = len(state)
                print(f"   Model: {num_layers} layers, ~{num_params/1e6:.1f}M params")
            
            # Check training state
            if 'optimizer_state' in ckpt:
                print(f"   Optimizer: {list(ckpt['optimizer_state'].keys())}")
            
            # Check epoch
            if 'epoch' in ckpt:
                print(f"   Epoch: {ckpt['epoch']}")
            
            # Check loss/metrics
            if 'best_val_loss' in ckpt:
                print(f"   Best val_loss: {ckpt['best_val_loss']:.4f}")
            if 'best_dice' in ckpt:
                print(f"   Best Dice: {ckpt['best_dice']:.4f}")
        
        print()
        
    except Exception as e:
        print(f"❌ {name}: ERROR {e}\n")

print("\n=== EXPECTED CONFIGURATIONS ===\n")
configs = {
    'stage1_binary_96': {
        'task': 'Binary segmentation (vessel vs background)',
        'num_classes': 2,
        'num_channels': 1,
        'input_size': '128x128x96',
        'feature_size': 24,
    },
    'stage2_level1_96': {
        'task': 'Family segmentation (8 groups)',
        'num_classes': 8,
        'num_channels': 2,
        'input_size': '128x128x64',
        'feature_size': 24,
    },
    'stage3_fine_96': {
        'task': 'Fine segmentation (41 classes)',
        'num_classes': 41,
        'num_channels': 2,
        'input_size': '128x128x64',
        'feature_size': 24,
    },
    'stage3_level2_v2': {
        'task': 'Fine segmentation v2 (41 classes)',
        'num_classes': 41,
        'num_channels': 2,
        'input_size': '128x128x64',
        'feature_size': 24,
    },
}

for name, config in configs.items():
    print(f"📌 {name}")
    for k, v in config.items():
        print(f"   {k}: {v}")
    print()
