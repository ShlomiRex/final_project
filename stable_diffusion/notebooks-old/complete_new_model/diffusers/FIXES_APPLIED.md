# Training Fixes Applied to train7.py

## Date: October 9, 2025

## Problem Analysis
The model was generating progressively worse images (green, blurry, lacking detail) due to:
1. Too small image dimensions leading to spatial information loss
2. Inconsistent VAE scaling factors
3. Insufficient training epochs
4. Lack of classifier-free guidance
5. Overly aggressive channel downsampling

## Fixes Applied

### 1. Image Size Increase
- **Before**: `IMAGE_SIZE = 32` → Latents: 8×8 → After 3 downsamplings: 1×1 (too small!)
- **After**: `IMAGE_SIZE = 64` → Latents: 16×16 → After 2 downsamplings: 4×4 (better spatial info)
- **Reason**: Larger spatial dimensions preserve more information through the network

### 2. Training Duration
- **Before**: `NUM_EPOCHS = 10`, `MAX_TRAIN_STEPS = 400_000`
- **After**: `NUM_EPOCHS = 100`, `MAX_TRAIN_STEPS = 10_000_000`
- **Reason**: Diffusion models require significantly more training to learn good representations

### 3. Checkpoint & Image Generation Frequency
- **Before**: Save every 5 epochs, generate every 1 epoch
- **After**: Save every 10 epochs, generate every 5 epochs
- **Reason**: With 100 epochs, less frequent saves prevent disk clutter while maintaining monitoring

### 4. UNet Architecture
- **Before**: `channel_mults=(1, 2, 4)` → 3 downsampling levels → [128, 256, 512] channels
- **After**: `channel_mults=(1, 2, 2)` → 2 downsampling levels → [128, 256, 256] channels
- **Reason**: Less aggressive downsampling for 64×64 images prevents spatial collapse

### 5. VAE Scaling Factor
- **Before**: Hardcoded `0.18215` (SD 1.x default)
- **After**: Uses actual config value with fallback `0.13025` (SDXL VAE)
- **Reason**: SDXL VAE uses different scaling - using wrong value corrupts latent space
- **Also Fixed**: Consistent scaling during both encode and decode

### 6. Classifier-Free Guidance Training
- **Added**: `CLASSIFIER_FREE_GUIDANCE_DROPOUT = 0.1`
- **Implementation**: 10% of training steps use unconditional (empty) text embeddings
- **Reason**: Enables better text-to-image alignment and allows CFG during inference

### 7. Inference VAE Decoding
- **Before**: Generic exception handling with fallback
- **After**: Uses consistent `vae_scaling_factor` from config, proper error messages
- **Reason**: Ensures decode uses same scaling as encode for consistent results

### 8. Loss Monitoring Improvements
- **Added**: Min/max loss tracking per epoch
- **Added**: Warning system for high loss values after initial training
- **Reason**: Early detection of training collapse or convergence issues

## Expected Improvements

1. **Better Spatial Information**: 16×16 latents instead of 8×8 preserve more detail
2. **Consistent Latent Space**: Proper VAE scaling prevents corruption
3. **Better Text Alignment**: CFG dropout improves conditioning
4. **More Training**: 10× more epochs allows better learning
5. **Reduced Collapse Risk**: Less aggressive downsampling maintains feature diversity

## What to Monitor During Training

1. **Loss values**: Should steadily decrease over first 20-30 epochs
2. **Generated images**: Should show progressive improvement every 5 epochs
3. **Color diversity**: Should move beyond single-color (green) outputs
4. **Shape formation**: Objects should become more recognizable over time
5. **Text alignment**: Generated images should match their text prompts better

## Next Steps if Issues Persist

1. **If loss plateaus but images stay bad**:
   - Reduce learning rate: `LEARNING_RATE = 5e-5`
   - Add learning rate scheduler (cosine annealing)

2. **If mode collapse continues** (all images look similar):
   - Increase CFG dropout: `CLASSIFIER_FREE_GUIDANCE_DROPOUT = 0.15`
   - Try different noise scheduler (scaled_linear beta schedule)

3. **If memory issues occur**:
   - Reduce `BATCH_SIZE` to 8 or 4
   - Increase `GRAD_ACCUM_STEPS` to compensate

4. **For faster experimentation**:
   - Can temporarily reduce to `IMAGE_SIZE = 32` with `channel_mults=(1, 2)` (only 1 downsampling)
   - Reduce `NUM_EPOCHS` to 20-30 for quick tests

## Training Command
```bash
python train7.py
```

## Notes
- First meaningful improvements should be visible around epoch 10-15
- High-quality results may require 50+ epochs
- Monitor GPU memory usage - 64×64 images use ~2× memory of 32×32
- The model is still training from scratch - don't expect stable diffusion quality without much more training and larger datasets
