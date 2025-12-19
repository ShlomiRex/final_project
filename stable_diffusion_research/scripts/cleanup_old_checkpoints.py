#!/usr/bin/env python3
"""
Clean up old incompatible checkpoints after switching to Diffusers UNet.

This script moves old checkpoints to a backup directory so training can start fresh.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def backup_old_checkpoints():
    """Move old checkpoints to backup directory."""
    
    # Checkpoint directories to check
    checkpoint_dirs = [
        Path("outputs/checkpoints"),
        Path("checkpoints"),
    ]
    
    all_checkpoint_dirs = []
    
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        
        # Find checkpoint-* subdirectories
        checkpoint_subdirs = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoint_subdirs:
            all_checkpoint_dirs.extend(checkpoint_subdirs)
    
    if not all_checkpoint_dirs:
        print(f"✅ No checkpoints found")
        print("   Training will start from scratch.")
        return
    
    print(f"Found {len(all_checkpoint_dirs)} checkpoint directories")
    for ckpt_dir in all_checkpoint_dirs:
        print(f"   - {ckpt_dir}")
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_base = Path("checkpoints_backup")
    backup_dir = backup_base / f"{timestamp}_custom_unet"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Backing up old checkpoints to: {backup_dir}")
    
    # Move each checkpoint directory
    for ckpt_dir in all_checkpoint_dirs:
        dest = backup_dir / ckpt_dir.name
        shutil.move(str(ckpt_dir), str(dest))
        print(f"   Moved: {ckpt_dir}")
    
    # Also move 'latest' file if exists
    for checkpoint_dir in checkpoint_dirs:
        latest_file = checkpoint_dir / "latest"
        if latest_file.exists():
            dest = backup_dir / f"latest_{checkpoint_dir.name}"
            shutil.move(str(latest_file), str(dest))
            print(f"   Moved: {latest_file}")
    
    print(f"\n✅ Backup complete!")
    print(f"   Old checkpoints saved to: {backup_dir}")
    print(f"   Training will now start from scratch with Diffusers UNet.\n")

if __name__ == "__main__":
    print("="*80)
    print(" Checkpoint Cleanup - Switching to Diffusers UNet")
    print("="*80)
    print("\nThis script will backup old checkpoints from the custom UNet architecture.")
    print("The new Diffusers UNet has a different architecture and cannot load them.\n")
    
    backup_old_checkpoints()
    
    print("\nNext steps:")
    print("1. Submit training job: sbatch slurm/train_2gpu.sh configs/base.yaml")
    print("2. Training will start from scratch with proper Diffusers architecture")
    print("="*80)
