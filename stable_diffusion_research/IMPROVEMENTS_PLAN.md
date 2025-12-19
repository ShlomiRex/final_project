# Improvements & MNIST Testing Plan

## ✅ MNIST Support - IMPLEMENTED

### What Changed:
1. **Dataset Loading**: Added MNIST dataset support in `src/data/dataset.py`
   - Automatically creates text prompts: `"a photograph of the number {0-9}"`
   - Uses MNIST's 28x28 grayscale images (upscaled to 256x256)
   - 60,000 training images (6,000 per digit)

2. **Config Updated**: Changed default dataset from "Matthijs/snacks" to "mnist"
   - Disabled `random_flip` (digits should maintain orientation)
   - Kept resolution at 256x256

### Why MNIST First?
- ✅ **Fast iteration**: 60K images vs millions (trains in hours, not days)
- ✅ **Clear success metric**: Can visually verify digit generation
- ✅ **Simple text-image alignment**: 10 distinct prompts only
- ✅ **Debugging**: Easy to spot if model learns text conditioning
- ✅ **Baseline**: If it fails on MNIST, it will fail on harder datasets

### Expected Results on MNIST:
- **After 1-2 hours (~5k steps)**: Should generate recognizable digit shapes
- **After 4-6 hours (~20k steps)**: Clear, sharp digits with correct text alignment
- **After 12 hours (~50k steps)**: High-quality digits, strong text-image correspondence

---

## 📊 CLIP & FID Score Metrics - PLAN

### Current State:
✅ **Already Implemented!** The codebase has:
- `src/evaluation/fid.py` - FID calculator using Inception-v3
- `src/evaluation/clip_score.py` - CLIP score calculator
- `src/evaluation/evaluator.py` - Orchestrates both metrics

### What's Missing: MLflow Integration

**TODO: Enable evaluation and log to MLflow**

#### Step 1: Enable Evaluation in Config
```yaml
# configs/base.yaml
evaluation:
  enabled: true  # ⚠️ Currently FALSE - change to TRUE
  eval_every_n_steps: 5000  # Evaluate every 5k steps
  
  # Sample Generation
  num_samples: 16
  sample_prompts:
    - "a photograph of the number 0"
    - "a photograph of the number 1"
    - "a photograph of the number 2"
    - "a photograph of the number 3"
    - "a photograph of the number 4"
    - "a photograph of the number 5"
    - "a photograph of the number 6"
    - "a photograph of the number 7"
    - "a photograph of the number 8"
    - "a photograph of the number 9"
  include_unconditional: false  # MNIST should be conditional only
  
  # FID Score
  fid:
    enabled: true
    num_samples: 1000  # Generate 1000 images for FID (reduce from 2048 for speed)
    batch_size: 64
  
  # CLIP Score
  clip_score:
    enabled: true
    num_samples: 500  # 500 images for CLIP score (reduce from default)
    batch_size: 64
```

#### Step 2: Update Trainer to Log Metrics
Modify `src/training/trainer.py` in the evaluation section to log to MLflow:

```python
# In trainer.py, after evaluation runs:
if self.mlflow_logger:
    # Log evaluation metrics
    self.mlflow_logger.log_metrics({
        "eval/fid_score": metrics["fid"],
        "eval/clip_score": metrics["clip_score"],
    }, step=self.global_step)
    
    # Log sample images
    self.mlflow_logger.log_images(
        samples,  # Dict of prompt -> image
        step=self.global_step,
        prefix="eval_samples"
    )
```

#### Step 3: Generate Reference Statistics for FID
For accurate FID on MNIST, we need reference statistics from real MNIST:

```python
# scripts/generate_mnist_fid_stats.py
from datasets import load_dataset
from src.evaluation.fid import FIDCalculator

# Load MNIST
mnist = load_dataset("mnist", split="test")

# Calculate statistics from real images
fid_calc = FIDCalculator()
fid_calc.calculate_and_save_reference_stats(
    real_images=[img["image"] for img in mnist],
    save_path="reference_stats/mnist_test_fid.npz"
)
```

Then update config:
```yaml
fid:
  enabled: true
  reference_stats: "reference_stats/mnist_test_fid.npz"
```

---

## 🚀 Code Improvements - RECOMMENDATIONS

### 1. **Remove Unnecessary Components** ❌

#### Remove Multi-Resolution Training (Not Needed for MNIST)
**File**: `src/data/dataset.py`
- Delete `MultiResolutionDataset` class (lines ~120-200)
- Delete `AspectRatioBucketDataset` class (lines ~200-350)
- **Reason**: MNIST is single resolution, adds complexity without benefit

#### Remove Offset Noise (Not Critical)
**File**: `src/training/trainer.py` (lines ~319-325)
```python
# Remove this:
if self.offset_noise > 0:
    offset = torch.randn(...)
    noise = noise + self.offset_noise * offset
```
**Reason**: Minor improvement for bright/dark images, not needed for grayscale MNIST

#### Remove MLflow Experiment Auto-Resume
**File**: `configs/base.yaml`
```yaml
checkpoint:
  resume_from_latest: false  # Set to FALSE by default
```
**Reason**: Prevents accidental loading of incompatible checkpoints (like we just saw)

### 2. **Add Important Features** ✅

#### Add Learning Rate Warmup Logging
**File**: `src/training/trainer.py`
```python
# In training loop, log current LR
if self.global_step % self.log_every_n_steps == 0:
    current_lr = self.optimizer.param_groups[0]['lr']
    if self.mlflow_logger:
        self.mlflow_logger.log_metric("train/learning_rate", current_lr, step=self.global_step)
```

#### Add Gradient Norm Logging
**File**: `src/training/trainer.py`
```python
# After gradient clipping
if self.accelerator.sync_gradients:
    grad_norm = self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
    if self.mlflow_logger and self.global_step % self.log_every_n_steps == 0:
        self.mlflow_logger.log_metric("train/grad_norm", grad_norm, step=self.global_step)
```

#### Add EMA Decay Logging
**File**: `scripts/train.py`
```python
# After MLflow initialization
if mlflow_logger:
    mlflow_logger.log_metric("model/ema_decay", ema_config.get("decay", 0.9999), step=0)
```

#### Add Sample Diversity Metric
Track how diverse the generated samples are:
```python
# In evaluator.py, add to calculate_metrics()
from torchvision.transforms import functional as F

def calculate_sample_diversity(images):
    """Calculate mean pairwise LPIPS distance"""
    # Use LPIPS for perceptual diversity
    # Higher = more diverse
    pass

metrics["sample_diversity"] = calculate_sample_diversity(fid_images)
```

### 3. **Improve MLflow Logging** 📈

#### Current Issues:
- ❌ No GPU memory tracking
- ❌ No training time per step
- ❌ No dataset info logged
- ❌ Sample images logged without prompts

#### Improvements:

**A. Add System Metrics**
```python
# In trainer.py, log every N steps
if self.global_step % 100 == 0:
    metrics = {
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "system/steps_per_second": steps_per_second,  # Calculate from timestamps
    }
    self.mlflow_logger.log_metrics(metrics, step=self.global_step)
```

**B. Log Dataset Statistics**
```python
# In train.py, after loading dataset
if mlflow_logger:
    mlflow_logger.log_params({
        "dataset/name": dataset_name,
        "dataset/size": len(train_dataloader.dataset),
        "dataset/num_batches": len(train_dataloader),
        "dataset/classes": len(class_names) if class_names else "N/A",
    })
```

**C. Improve Image Logging**
```python
# Log samples with prompts as titles
mlflow_logger.log_images(
    images=samples,  # Dict[prompt, image]
    step=global_step,
    prefix="samples",
    labels=list(samples.keys()),  # Add prompts as labels
)
```

**D. Add Training Progress Metrics**
```python
# Track ETA and progress
metrics = {
    "progress/epoch": epoch,
    "progress/steps_completed": global_step,
    "progress/steps_remaining": max_steps - global_step,
    "progress/percent_complete": 100 * global_step / max_steps,
}
```

### 4. **Config Improvements** ⚙️

#### Simplify for MNIST
**File**: `configs/base.yaml`

```yaml
# Reduce logging spam
logging:
  mlflow:
    log_every_n_steps: 50  # Log every 50 steps (was 10)
  console:
    log_every_n_steps: 100  # Reduce console spam

# Faster checkpointing for MNIST
checkpoint:
  save_every_n_steps: 2000  # More frequent (was 5000)
  keep_last_n: 3  # Only keep 3 checkpoints (was 5)

# Simpler evaluation
evaluation:
  enabled: true
  eval_every_n_steps: 2000  # Align with checkpoints
  num_samples: 10  # One per digit
```

### 5. **Model Architecture Improvements** 🧠

#### Current Config (from base.yaml):
```yaml
unet:
  model_channels: 320
  channel_mult: [1, 2, 4, 4]  # -> [320, 640, 1280, 1280]
```

#### For MNIST - Consider Smaller Model:
```yaml
# configs/mnist.yaml (NEW)
model:
  unet:
    model_channels: 128  # Reduce from 320
    channel_mult: [1, 2, 4, 4]  # -> [128, 256, 512, 512]
    # Result: ~200M params instead of ~860M
```

**Reasoning**:
- MNIST is simple (10 classes, grayscale)
- Smaller model = faster training + less overfitting
- Can always scale up after MNIST validation

---

## 🎯 Priority Action Items

### Immediate (Before Next Training Run):
1. ✅ **DONE**: MNIST dataset integration
2. ⚠️ **TODO**: Enable evaluation in config (`enabled: true`)
3. ⚠️ **TODO**: Update sample prompts to MNIST digits
4. ⚠️ **TODO**: Reduce batch size to 32 if OOM (currently 64)

### High Priority (Within 24 Hours):
5. ⚠️ **TODO**: Add learning rate & gradient norm logging
6. ⚠️ **TODO**: Add system metrics (GPU memory, speed)
7. ⚠️ **TODO**: Generate MNIST reference statistics for FID
8. ⚠️ **TODO**: Create `configs/mnist.yaml` with optimized settings

### Medium Priority (This Week):
9. Remove multi-resolution code (cleanup)
10. Add sample diversity metric
11. Improve image logging with prompts
12. Add training progress tracking

### Low Priority (Nice to Have):
13. Remove offset noise
14. Add automated hyperparameter tuning
15. Add Weights & Biases integration (alternative to MLflow)

---

## 📝 Recommended Training Workflow for MNIST

### Step 1: Quick Sanity Check (1 hour)
```bash
# Update config to use MNIST (already done)
# Run short training
sbatch slurm/train_2gpu.sh configs/base.yaml

# After 30 min (~2k steps):
# - Check MLflow: loss should be ~0.05-0.1
# - Check samples: should see vague digit shapes
# - If working, continue to Step 2
```

### Step 2: Full MNIST Training (12 hours)
```bash
# Continue training or start fresh
sbatch slurm/train_2gpu.sh configs/base.yaml

# After 12 hours (~50k steps):
# - Loss: ~0.01-0.02
# - FID: <50 (lower is better)
# - CLIP Score: >0.25 (higher is better)
# - Visual: Clear, sharp digits
```

### Step 3: Evaluation & Analysis
```python
# Generate test samples
python scripts/generate.py --checkpoint outputs/checkpoints/checkpoint-00050000 \
    --prompt "a photograph of the number 7" --num_samples 16

# Check MLflow dashboard
# - Compare FID across checkpoints
# - Visualize CLIP score progression
# - Identify best checkpoint
```

### Step 4: Scale to Real Dataset
After MNIST validation:
```bash
# Switch to larger dataset
# Update config: dataset: "coco" or "laion"
# Train with proven architecture
```

---

## 🔍 Debugging Checklist

If MNIST training fails:

### Loss Not Decreasing:
- [ ] Check learning rate (should be ~1e-4)
- [ ] Verify CFG is working (10% null conditioning)
- [ ] Check noise scheduler (cosine)
- [ ] Verify text prompts are being used

### Generated Images Are Blobs:
- [ ] Check VAE scaling (0.18215)
- [ ] Verify UNet architecture (Diffusers)
- [ ] Check null embedding (not zeros)
- [ ] Increase training steps

### OOM Errors:
- [ ] Reduce batch_size from 64 to 32
- [ ] Enable gradient checkpointing
- [ ] Reduce model size (128 base channels)

### Training Too Slow:
- [ ] Check GPU utilization (should be >90%)
- [ ] Reduce num_workers if CPU bottleneck
- [ ] Disable evaluation during debugging

---

## 📊 Expected MLflow Dashboard

After implementation, MLflow should show:

### Metrics Tab:
- `train/loss` - Training loss curve
- `train/learning_rate` - LR warmup & decay
- `train/grad_norm` - Gradient magnitude
- `eval/fid_score` - FID score over time
- `eval/clip_score` - CLIP score over time
- `system/gpu_memory_allocated_gb` - Memory usage
- `system/steps_per_second` - Training speed
- `model/total_parameters_millions` - 860M (Diffusers UNet)

### Images Tab:
- `eval_samples/step_5000` - Generated digits at 5k steps
- `eval_samples/step_10000` - Generated digits at 10k steps
- ...

### Parameters Tab:
- `dataset/name`: "mnist"
- `model/unet/model_channels`: 320
- `training/batch_size`: 64
- `training/learning_rate`: 1e-4

---

## Summary

**What's Done**: ✅
- MNIST dataset support with text prompts
- Model architecture (Diffusers UNet)
- CFG training with proper null embeddings
- Cosine noise scheduler
- Multi-GPU support

**What's Ready (Just Enable)**: 🟡
- FID score calculation
- CLIP score calculation
- Sample generation
- Checkpoint management

**What Needs Implementation**: ⚠️
- MLflow integration for eval metrics
- System metrics logging (GPU, speed)
- Reference statistics for FID
- MNIST-specific config optimization

**Recommended Next Steps**:
1. Enable evaluation in config
2. Run 1-hour MNIST sanity check
3. Add MLflow eval metric logging
4. Generate MNIST reference stats
5. Run full 12-hour MNIST training
6. Analyze results and tune hyperparameters
