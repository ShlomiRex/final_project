# Train10 Notebook Changes: LSUN Churches → MS-COCO Text-to-Image

## Summary of Changes

The `train10_hpc_celeba_diffusers.ipynb` notebook has been completely transformed from an **unconditional** church image generator to a **text-conditional** MS-COCO text-to-image model.

---

## Major Changes

### 1. **Dataset**
- **Before**: LSUN Churches (~126K church images, unconditional)
- **After**: MS-COCO (~120K diverse images with text captions)
- **Implementation**: Streaming dataset via HuggingFace to save disk space

### 2. **Model Architecture**
- **Before**: `UNet2DModel` (unconditional, no cross-attention)
- **After**: `UNet2DConditionModel` (cross-attention layers for text conditioning)
- **Added**: CLIP text encoder and tokenizer (frozen, pretrained)

### 3. **Training Configuration**
- **Epochs**: 1 → **500 epochs**
- **Output directory**: `./outputs/train10_lsun_churches` → `./outputs/train10_coco_text2img`
- **Classifier-free guidance**: Added 10% unconditional training for CFG
- **Streaming**: Enabled to save disk space

### 4. **Training Loop**
- **Before**: Simple noise prediction without conditioning
- **After**: Text-conditional noise prediction with cross-attention
- **Added**: Text encoding step for each batch
- **Added**: Support for classifier-free guidance during training

### 5. **Sampling/Generation**
- **Before**: Unconditional random generation
- **After**: Text-to-image generation from prompts
- **Added**: Classifier-free guidance scale parameter (default: 7.5)
- **Added**: Custom prompt support

### 6. **Sample Visualization**
- **Before**: Random unconditional church images
- **After**: Images generated from predefined text prompts
- **Examples**: "A cat on a couch", "A person riding a bicycle", etc.

---

## Technical Details

### New Imports
```python
from diffusers.models import UNet2DConditionModel  # Was: UNet2DModel
from transformers import CLIPTextModel, CLIPTokenizer  # New
```

### New Helper Function
```python
def encode_text(text: str, tokenizer, text_encoder, device):
    """Encode text prompt into embeddings using CLIP."""
    # Returns: text_embeddings tensor
```

### Model Updates
- **VAE**: Still frozen, unchanged
- **CLIP Text Encoder**: Added, frozen (openai/clip-vit-base-patch32)
- **UNet**: Changed to conditional variant with cross-attention
  - Cross-attention dimension: 512 (CLIP hidden size)
  - 4 down blocks (3 with cross-attention)
  - 4 up blocks (3 with cross-attention)

### Dataset Format
```python
# COCO batch structure:
{
    'image': tensor,  # Transformed image
    'caption': str,   # Random caption from 5 available
}
```

### Training Parameters
- **Batch size**: 16
- **Learning rate**: 1e-4
- **Image size**: 256x256
- **Mixed precision**: FP16
- **Checkpoint interval**: Every 2000 steps
- **Total epochs**: 500
- **Estimated total steps**: ~3.75 million

---

## Usage Examples

### Generate from Custom Prompt
```python
generated_image = sample(
    prompt="A golden retriever playing in a park",
    num_inference_steps=50,
    guidance_scale=7.5,  # Higher = more prompt adherence
    seed=42
)
```

### Training
```python
# Automatic checkpoint resumption
batch_losses, epoch_losses = train(
    config, vae, text_encoder, tokenizer, unet, dataloader, device,
    resume_from_checkpoint=latest_checkpoint
)
```

---

## Key Features

✅ **Text-to-Image Generation**: Generate images from natural language prompts  
✅ **Classifier-Free Guidance**: Better prompt adherence  
✅ **Checkpoint System**: Automatic resume with validation  
✅ **Streaming Dataset**: No need to download entire COCO dataset  
✅ **High-Quality Captions**: 5 human-written captions per image  
✅ **Long Training**: 500 epochs for thorough learning  
✅ **Mixed Precision**: FP16 for memory efficiency  

---

## Expected Results

After 500 epochs of training (~3.75M steps):
- Model should generate coherent images from text prompts
- Quality will depend on prompt complexity
- Checkpoint every 2000 steps = ~1,875 checkpoints
- Can monitor progress through sample generations

---

## Next Steps

1. Run the notebook cells in order
2. Monitor checkpoint validation images
3. Adjust `guidance_scale` for generation quality (7.5 is default)
4. Modify prompts in the generation cells for custom outputs
5. Training will take several days/weeks on single GPU for 500 epochs
