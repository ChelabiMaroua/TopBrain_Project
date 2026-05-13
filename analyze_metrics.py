#!/usr/bin/env python3
"""Quick analysis of training metrics."""
import json
from pathlib import Path

files = [
    '5_HierarchicalSeg/checkpoints/stage2_level1_96/history_level1_fold_1.json',
    '5_HierarchicalSeg/checkpoints/stage3_fine_96/history_level2_fold_1.json',
    '5_HierarchicalSeg/checkpoints/stage3_fine_v1/history_level2_fold_1.json',
    '5_HierarchicalSeg/checkpoints/stage3_level2_v2/history_level2_fold_1.json'
]

print("\n=== TRAINING METRICS SUMMARY ===\n")

for f in files:
    p = Path(f)
    try:
        with open(p) as j:
            data = json.load(j)
        
        epochs = len(data)
        first = data[0] if data else {}
        last = data[-1] if data else {}
        
        # Find best epoch
        best_epoch = max(range(len(data)), key=lambda i: data[i].get('val_dice_mean', -1))
        best = data[best_epoch]
        
        print(f"📊 {p.parent.name}/")
        print(f"  Epochs trained: {epochs}")
        print(f"  Best epoch: {best.get('epoch', best_epoch+1)}")
        
        if 'val_dice_mean' in last:
            print(f"  Final val_dice: {last['val_dice_mean']:.4f}")
        if 'val_dice_mean' in best:
            print(f"  Best val_dice:  {best['val_dice_mean']:.4f}")
        if 'val_loss' in last:
            print(f"  Final val_loss: {last['val_loss']:.4f}")
        
        train_time = last.get('total_hours', 0)
        print(f"  Training time: {train_time:.2f} hours")
        print()
        
    except Exception as e:
        print(f"❌ {f}: {e}\n")

print("\n=== CHECKPOINT SIZES ===\n")
ckpt_dir = Path('5_HierarchicalSeg/checkpoints')
for pth in ckpt_dir.rglob('*.pth'):
    size_mb = pth.stat().st_size / (1024**2)
    print(f"  {pth.relative_to(ckpt_dir)}: {size_mb:.1f} MB")
