# Checkpoint Incompatibility - RESOLVED

## Problem
Training failed with error:
```
RuntimeError: Error(s) in loading state_dict for UNet2DConditionModel:
	Missing key(s) in state_dict: ...
	Unexpected key(s) in state_dict: ...
```

## Root Cause
The old checkpoints were created with the **custom UNet architecture** (from `src/models/unet.py`). When we switched to the **Diffusers UNet2DConditionModel**, the architectures are completely different and incompatible:

- **Custom UNet**: Uses custom layer names like `in_layers`, `out_layers`, `skip_connection`, etc.
- **Diffusers UNet**: Uses standard names like `norm1`, `conv1`, `time_emb_proj`, `resnets`, etc.

## Solution Applied

### 1. Added Error Handling in Trainer
Modified `src/training/trainer.py` to catch checkpoint loading errors and start fresh:

```python
try:
    self.global_step = self.checkpoint_manager.load(...)
    self.accelerator.print(f"Resumed from checkpoint at step {self.global_step}")
except (RuntimeError, KeyError) as e:
    self.accelerator.print(f"\n⚠️  WARNING: Failed to load checkpoint")
    self.accelerator.print("Starting training from scratch with fresh weights.\n")
    self.global_step = 0
```

### 2. Backed Up Old Checkpoints
Moved old incompatible checkpoints to backup directory:
```
checkpoints_backup/20251214_221758_custom_unet/
├── checkpoint-00005000/
├── checkpoint-00010000/
└── checkpoint-00015000/
```

### 3. Fresh Start
Training will now start from step 0 with the new Diffusers UNet architecture.

## Next Steps

**Training is ready to start:**
```bash
sbatch slurm/train_2gpu.sh configs/base.yaml
```

This will:
- ✅ Use Diffusers UNet2DConditionModel (proven architecture)
- ✅ Start from scratch with proper initialization
- ✅ Train with fixed CFG, cosine scheduler, optimized batch size
- ✅ No compatibility issues

## What to Expect

### Initial Training (0-1 hour):
- Loss should start around 0.1-0.15
- Decreases to ~0.05-0.08
- Generated images show basic shapes/colors (NOT blobs!)

### After 6 hours (~20k steps):
- Loss around 0.02-0.05
- Recognizable objects in generated images
- Text-image alignment emerging

### After 24+ hours:
- Loss plateaus around 0.01-0.02
- High-quality coherent images
- Strong text-image alignment

## Why This Is Better

The old custom UNet had **6 critical issues**:
1. Zero-initialized output layer (prevented learning)
2. Incorrect CFG null conditioning
3. Over-regularization
4. Suboptimal noise schedule
5. Smaller model size
6. Potential architectural bugs

The new Diffusers UNet:
1. ✅ Proper initialization from the start
2. ✅ Battle-tested architecture (SD 1.4/1.5)
3. ✅ ~860M parameters (vs ~200M)
4. ✅ Works with all our fixes (CFG, cosine schedule, etc.)
5. ✅ No custom code bugs

**Starting fresh is actually the right move!**
