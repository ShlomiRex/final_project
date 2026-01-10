#!/usr/bin/env python3
"""
Validation script to verify all fixes are correctly applied.

Run this before training to ensure all changes are in place.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check that Diffusers UNet can be imported."""
    print("✓ Checking imports...")
    try:
        from diffusers import UNet2DConditionModel
        print("  ✅ Diffusers UNet2DConditionModel import successful")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import Diffusers: {e}")
        return False


def check_config():
    """Check configuration file has correct values."""
    print("\n✓ Checking configuration...")
    import yaml
    
    config_path = Path(__file__).parent / "configs" / "base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    issues = []
    
    # Check batch size
    batch_size = config.get("data", {}).get("batch_size")
    if batch_size == 64:
        print(f"  ✅ Batch size: {batch_size}")
    else:
        print(f"  ❌ Batch size should be 64, got: {batch_size}")
        issues.append("batch_size")
    
    # Check noise schedule
    beta_schedule = config.get("diffusion", {}).get("beta_schedule")
    if beta_schedule == "squaredcos_cap_v2":
        print(f"  ✅ Noise schedule: {beta_schedule}")
    else:
        print(f"  ❌ Noise schedule should be 'squaredcos_cap_v2', got: {beta_schedule}")
        issues.append("beta_schedule")
    
    # Check weight decay
    weight_decay = config.get("training", {}).get("optimizer", {}).get("weight_decay")
    if weight_decay == 0.0:
        print(f"  ✅ Weight decay: {weight_decay}")
    else:
        print(f"  ⚠️  Weight decay: {weight_decay} (should be 0.0 but might work)")
    
    # Check CFG
    cfg_enabled = config.get("training", {}).get("cfg", {}).get("enabled")
    uncond_prob = config.get("training", {}).get("cfg", {}).get("uncond_prob")
    if cfg_enabled:
        print(f"  ✅ CFG enabled with uncond_prob: {uncond_prob}")
    else:
        print(f"  ❌ CFG should be enabled")
        issues.append("cfg_enabled")
    
    return len(issues) == 0


def check_trainer_code():
    """Check that trainer code has null embedding fix."""
    print("\n✓ Checking trainer code...")
    
    trainer_path = Path(__file__).parent / "src" / "training" / "trainer.py"
    with open(trainer_path) as f:
        content = f.read()
    
    # Check for null embedding usage
    if "get_null_embedding" in content:
        print("  ✅ CFG uses null embedding (get_null_embedding)")
    else:
        print("  ❌ CFG null embedding fix not found")
        return False
    
    # Check for incorrect zero assignment
    if "encoder_hidden_states[drop_mask] = 0" in content:
        print("  ❌ Found old CFG code (zeroing embeddings)")
        return False
    else:
        print("  ✅ No old CFG code found (not zeroing embeddings)")
    
    return True


def check_train_script():
    """Check that train.py uses Diffusers UNet."""
    print("\n✓ Checking train.py...")
    
    train_path = Path(__file__).parent / "scripts" / "train.py"
    with open(train_path) as f:
        content = f.read()
    
    # Check for Diffusers import
    if "from diffusers import UNet2DConditionModel" in content:
        print("  ✅ Using Diffusers UNet2DConditionModel")
    else:
        print("  ❌ Not using Diffusers UNet2DConditionModel")
        return False
    
    # Check for custom UNet import (should NOT be present)
    if "from src.models.unet import UNet2DConditionModel" in content:
        print("  ❌ Still importing custom UNet")
        return False
    else:
        print("  ✅ Not using custom UNet")
    
    return True


def check_accelerate_config():
    """Check accelerate config file exists."""
    print("\n✓ Checking Accelerate config...")
    
    config_path = Path(__file__).parent / "accelerate_config.yaml"
    if config_path.exists():
        print(f"  ✅ Accelerate config exists: {config_path}")
        
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        num_processes = config.get("num_processes", 0)
        mixed_precision = config.get("mixed_precision", "no")
        
        print(f"     - num_processes: {num_processes}")
        print(f"     - mixed_precision: {mixed_precision}")
        
        return True
    else:
        print(f"  ❌ Accelerate config not found: {config_path}")
        return False


def main():
    print("=" * 80)
    print("STABLE DIFFUSION FIXES VALIDATION")
    print("=" * 80)
    
    all_passed = True
    
    # Run all checks
    all_passed &= check_imports()
    all_passed &= check_config()
    all_passed &= check_trainer_code()
    all_passed &= check_train_script()
    all_passed &= check_accelerate_config()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to train!")
        print("\nTo start training:")
        print("  sbatch slurm/train_2gpu.sh configs/base.yaml")
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
        print("\nRefer to FIXES_IMPLEMENTED.md for details on required changes")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
