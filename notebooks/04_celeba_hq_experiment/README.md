# Experiment 4: CelebA-HQ Latent Diffusion with Classifier-Free Guidance

## Overview

Experiment 4 extends the project to **latent diffusion** using CelebA-HQ dataset. Unlike previous experiments that operated in pixel space, this experiment trains the diffusion U-Net in VAE latent space for significantly faster training and higher resolution (256×256).

## Architecture

### Key Components

1. **VAE (Frozen, Pretrained)**
   - Model: `stabilityai/sd-vae-ft-mse`
   - Purpose: Encode images (256×256) → latents (32×32×4)
   - Compression: 8x spatial, 3→4 channels
   - Status: Frozen during training
   - Location: [models/vae_wrapper.py](../models/vae_wrapper.py)

2. **UNet (Trainable)**
   - Input/Output: 32×32×4 latents
   - Cross-attention: CLIP text embeddings (512-dim)
   - Parameters: ~90M (between CIFAR-10 and WikiArt)
   - Location: [models/custom_unet_celeba_ldm.py](../models/custom_unet_celeba_ldm.py)

3. **CLIP Text Encoder (Frozen)**
   - Model: `openai/clip-vit-base-patch32`
   - Output: 512-dimensional embeddings
   - Status: Frozen during training

4. **Noise Scheduler**
   - Type: DDPM
   - Schedule: Squared cosine
   - Timesteps: 1000

### Why Latent Diffusion?

| Aspect | Pixel Space | Latent Space |
|--------|-------------|--------------|
| Resolution | 256×256 | 32×32 |
| Channels | 3 (RGB) | 4 (VAE latents) |
| Memory | High | Low (64x less) |
| Speed | Slow | Fast (64x faster) |
| Quality | Good | Better (VAE denoising) |

## Dataset

### CelebA-HQ

- **Images**: 30,000 high-quality face images
- **Resolution**: 1024×1024 (resized to 256×256)
- **Attributes**: 40 binary attributes per image
  - Gender (Male/Female)
  - Age (Young/Older)
  - Hair color/style
  - Accessories (glasses, hat)
  - Facial features

### Text Captions

Captions are automatically generated from attributes:

```python
# Examples:
"A photo of a young woman with blond hair, smiling"
"A portrait of an older man with eyeglasses"
"A young person with black hair, wearing a hat"
```

Location: [custom_datasets/celeba_hq_dataset.py](../custom_datasets/celeba_hq_dataset.py)

## Training

### Configuration

```python
# From config.py
TRAIN_CELEBA_LDM_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-5,
    "batch_size": 32,
    "num_train_timesteps": 1000,
    "beta_schedule": "squaredcos_cap_v2",
    "checkpoint_every_n_epochs": 10,
    "cfg_dropout_prob": 0.1,  # 10% unconditional
}
```

### Training Pipeline

1. **Load image** (1024×1024 or varied)
2. **Resize + crop** → 256×256
3. **Normalize** → [-1, 1]
4. **Encode with VAE** → 32×32×4 latents
5. **Add noise** to latents
6. **Get text embeddings** from CLIP
7. **Apply CFG dropout** (10% empty prompts)
8. **Predict noise** with UNet
9. **Compute MSE loss**
10. **Optimize**

### Key Differences from Pixel-Space Training

```python
# Pixel-space (Experiments 1-3)
images = load_image()  # (B, C, H, W)
noisy_images = add_noise(images)
noise_pred = unet(noisy_images, t, text_embeddings)
loss = mse_loss(noise_pred, noise)

# Latent-space (Experiment 4)
images = load_image()  # (B, 3, 256, 256)
latents = vae.encode(images)  # (B, 4, 32, 32) ← VAE encoding
noisy_latents = add_noise(latents)  # Noise in latent space
noise_pred = unet(noisy_latents, t, text_embeddings)
loss = mse_loss(noise_pred, noise)
```

### Notebook

[train1_t2i_celeba_hq_ldm_cfg.ipynb](train1_t2i_celeba_hq_ldm_cfg.ipynb)

## Inference

### Sampling Pipeline

1. **Start with random latent noise** (B, 4, 32, 32)
2. **Denoise with CFG** (same as experiments 1-3)
3. **Decode with VAE** → (B, 3, 256, 256)
4. **Convert to [0, 1]** and save

### Classifier-Free Guidance

Same implementation as previous experiments:

```python
# Concatenate unconditional + conditional
latents_input = torch.cat([latents, latents])
text_embeddings_input = torch.cat([uncond_embeddings, cond_embeddings])

# Predict
noise_pred = unet(latents_input, t, text_embeddings_input)

# Split and combine with guidance scale
noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

### Notebook

**TODO**: `inference1_t2i_celeba_hq_ldm_cfg.ipynb`

## Evaluation

### Metrics

1. **FID (Fréchet Inception Distance)**
   - Compare real vs generated images
   - Use Inception features
   - Same as experiments 2-3

2. **CLIP Score**
   - Measure text-image alignment
   - Compute cosine similarity between CLIP embeddings
   - Industry standard for text-to-image models

3. **Attribute Consistency**
   - Train attribute classifier (ResNet-18 multi-label)
   - Check if generated images match requested attributes
   - E.g., "young woman with glasses" → verify predictions

### Notebooks

- **TODO**: `metrics1_evaluate_celeba_hq.ipynb`
- **TODO**: `train2_train_celeba_attribute_classifier.ipynb`

## File Structure

```
experiment 4/
├── train1_t2i_celeba_hq_ldm_cfg.ipynb         # ✓ Training
├── inference1_t2i_celeba_hq_ldm_cfg.ipynb     # TODO: Inference
├── metrics1_evaluate_celeba_hq.ipynb          # TODO: Metrics
└── train2_train_celeba_attribute_classifier.ipynb  # TODO: Classifier

Project root:
├── config.py                                   # ✓ Updated
├── models/
│   ├── vae_wrapper.py                         # ✓ VAE wrapper
│   └── custom_unet_celeba_ldm.py              # ✓ Latent UNet
├── custom_datasets/
│   └── celeba_hq_dataset.py                   # ✓ Dataset loader
└── checkpoints/
    └── celeba_ldm_unet_checkpoint_epoch_*.pt  # Training saves here
```

## Next Steps

### Immediate (for running training)

1. **Install CelebA-HQ dataset**:
   - Try: `tglcourse/CelebA-HQ-img` on HuggingFace
   - Or: `huggan/CelebA-HQ`, `nielsr/CelebA-HQ`
   - Adjust dataset name in training notebook if needed

2. **Run training notebook**: `train1_t2i_celeba_hq_ldm_cfg.ipynb`

### Future Development

3. **Create inference notebook** for generating images
4. **Create metrics notebook** for FID + CLIP score
5. **Train attribute classifier** for prompt adherence
6. **Optional**: Implement latent caching for faster training

## Comparison with Previous Experiments

| Experiment | Dataset | Resolution | Space | Channels | Speed | Params |
|------------|---------|------------|-------|----------|-------|---------|
| 1 (MNIST) | MNIST | 28×28 | Pixel | 1 | Fast | 2.6M |
| 2 (CIFAR-10) | CIFAR-10 | 32×32 | Pixel | 3 | Medium | ~20M |
| 3 (WikiArt) | WikiArt | 128×128 | Pixel | 3 | Slow | ~80M |
| **4 (CelebA)** | **CelebA-HQ** | **256×256** | **Latent** | **4** | **Fast** | **~90M** |

## Key Innovations

1. **Latent Diffusion**: First experiment using VAE-based latent space
2. **Higher Resolution**: 256×256 (2x larger than WikiArt in pixels)
3. **Pretrained VAE**: Leverages Stable Diffusion VAE
4. **Attribute-Based Prompts**: Generates captions from structured attributes
5. **CLIP Score**: Adds quantitative text-image alignment metric

## References

- **Latent Diffusion Models**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
- **CelebA-HQ**: Karras et al., "Progressive Growing of GANs", ICLR 2018
- **Stable Diffusion VAE**: StabilityAI, `stabilityai/sd-vae-ft-mse`
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
