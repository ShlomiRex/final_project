# AI Coding Agent Instructions

## Project Overview

**Autoregressive Latent Transformer for Text-to-Image Generation**

Train a GPT-style transformer (~500M parameters) that generates VQ-VAE latent tokens conditioned on CLIP text embeddings. The model supports both text-conditional and unconditional generation at multiple resolutions.

## Architecture Overview

```
Text Prompt → CLIP Encoder → Cross-Attention
                                    ↓
Image → VQ-VAE Encoder → Discrete Tokens → Autoregressive Transformer → Tokens → VQ-VAE Decoder → Image
```

### Core Components

| Component | Source | Trainable | Notes |
|-----------|--------|-----------|-------|
| **VQ-VAE** | Pretrained or custom | Phase-dependent | Encodes images to discrete tokens |
| **Text Encoder** | `openai/clip-vit-base-patch32` | ❌ Frozen | Hidden size: 512 |
| **Transformer** | Custom GPT-2 style | ✅ Yes | ~500M params, 24 layers |

### Model Configuration

- **Transformer**: 24 layers, 1024 hidden dim, 16 attention heads
- **VQ-VAE**: f=16 downsampling, 16384 codebook entries (configurable)
- **Sequence lengths**: 256 tokens (256×256 images) to 4096 tokens (512×512 with f=8)
- **Conditioning**: Cross-attention to CLIP embeddings or unconditional with learned null embedding

## Project Structure

```
/home/doshlom4/work/final_project/
├── .github/
│   └── copilot-instructions.md
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vqvae.py              # VQ-VAE wrapper (pretrained + custom)
│   │   ├── latent_gpt.py         # Autoregressive transformer
│   │   └── clip_encoder.py       # CLIP text encoder wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── flickr30k.py          # Flickr30k dataset
│   │   └── transforms.py         # Image transforms
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_vqvae.py        # VQ-VAE training script
│   │   └── train_transformer.py  # Transformer training script
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration classes
│       └── logging.py            # MLflow logging utilities
├── scripts/
│   ├── train_latent_gpt.py       # Main training entry point
│   └── inference.py              # Generation script
├── slurm/
│   ├── mlflow_server.sh          # MLflow server job
│   ├── train_8gpu.sh             # 8×A100 training job
│   └── README.md                 # Slurm usage guide
├── notebooks/
│   └── experiment_pretrained.ipynb
├── configs/
│   ├── base.yaml                 # Base configuration
│   ├── vqvae_pretrained.yaml     # Pretrained VQ-VAE config
│   └── transformer_500m.yaml     # 500M transformer config
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Training Phases

### Phase 1: Pretrained VQ-VAE Experimentation
1. Use pretrained VQGAN (`dalle-mini/vqgan_imagenet_f16_16384`)
2. Train small transformer to validate pipeline
3. Experiment with generation quality
4. **MLflow Experiment**: `latent-gpt-pretrained-vqvae`

### Phase 2: Custom VQ-VAE Training
1. Train VQ-VAE from scratch on Flickr30k
2. Optimize for domain-specific reconstruction
3. **MLflow Experiment**: `vqvae-training`

### Phase 3: Full Transformer Training
1. Train 500M transformer with custom VQ-VAE
2. Full-scale multi-GPU training
3. **MLflow Experiment**: `latent-gpt-custom-vqvae`

## MLflow Experiment Organization

### Experiments
| Experiment Name | Purpose | Key Metrics |
|-----------------|---------|-------------|
| `vqvae-training` | VQ-VAE from scratch | recon_loss, codebook_usage, perplexity |
| `latent-gpt-pretrained-vqvae` | Transformer with pretrained VQ-VAE | ce_loss, perplexity, FID |
| `latent-gpt-custom-vqvae` | Transformer with custom VQ-VAE | ce_loss, perplexity, FID, CLIP-score |

### Run Tags
- `resolution`: 256, 512
- `conditioning`: text_conditional, unconditional
- `vqvae_source`: pretrained_hf, pretrained_taming, custom
- `phase`: 1, 2, 3

### Logged Parameters
```python
mlflow.log_params({
    # VQ-VAE config
    "vqvae_source": "pretrained_hf",
    "vqvae_checkpoint": "dalle-mini/vqgan_imagenet_f16_16384",
    "codebook_size": 16384,
    "downsample_factor": 16,
    
    # Transformer config
    "transformer_layers": 24,
    "transformer_hidden": 1024,
    "transformer_heads": 16,
    "transformer_params": "500M",
    
    # Training config
    "batch_size": 32,
    "learning_rate": 1e-4,
    "resolution": 256,
    "conditioning": "text_conditional",
})
```

## HPC Cluster & Slurm

### Available Resources
- **GPU Nodes**: gpu6 (2×A100), gpu8 (8×A100 80GB)
- **Login Node**: `login9` (use for SSH tunneling)
- **Max Runtime**: 72 hours per job
- **Framework**: HuggingFace Accelerate for multi-GPU

### Python Environment
- **Location**: `/home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/`
- **Python**: 3.9.21
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **Activation**: `source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate`

### Accessing Servers from Local Machine

**CRITICAL**: Use `login9` as the SSH gateway (NOT `hpc-login9`).

When running servers on GPU nodes (Jupyter, MLflow, TensorBoard, etc.), use SSH port forwarding to access them from your local machine:

```bash
# Generic pattern for any server
ssh -N -L localhost:<LOCAL_PORT>:<GPU_NODE>.hpc.pub.lan:<REMOTE_PORT> doshlom4@login9

# Examples:
# Jupyter Lab on gpu8:8888
ssh -N -L localhost:8888:gpu8.hpc.pub.lan:8888 doshlom4@login9

# MLflow on gpu8:5000
ssh -N -L localhost:5000:gpu8.hpc.pub.lan:5000 doshlom4@login9

# TensorBoard on gpu6:6006
ssh -N -L localhost:6006:gpu6.hpc.pub.lan:6006 doshlom4@login9
```

Then access in browser: `http://localhost:<LOCAL_PORT>`

**Notes:**
- The `-N` flag prevents command execution (tunnel only)
- Keep the SSH tunnel running while accessing the server
- Press `Ctrl+C` to close the tunnel
- You can map to different local ports if needed: `-L localhost:9999:gpu8.hpc.pub.lan:8888`

### Multi-GPU Training
**CRITICAL**: Use Python scripts, NOT Jupyter notebooks for multi-GPU training.

```bash
# Launch with Accelerate
accelerate launch --num_processes=8 --multi_gpu --mixed_precision=bf16 \
    scripts/train_latent_gpt.py --config configs/transformer_500m.yaml
```

### MLflow Server on Slurm
```bash
# Submit MLflow server job
sbatch slurm/mlflow_server.sh

# Get allocated node
squeue -u $USER | grep mlflow

# Set tracking URI in training scripts
export MLFLOW_TRACKING_URI=http://<mlflow_node>:5000

# Access MLflow UI from local machine
ssh -N -L localhost:5000:<mlflow_node>.hpc.pub.lan:5000 doshlom4@login9
# Then open: http://localhost:5000
```

## Code Patterns

### MLflow Logging
```python
import mlflow

# Set experiment based on phase
mlflow.set_experiment("latent-gpt-pretrained-vqvae")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.set_tags({
        "resolution": "256",
        "conditioning": "text_conditional",
        "vqvae_source": "pretrained_hf",
        "phase": "1",
    })
    mlflow.log_params(config.__dict__)
    
    # Training loop
    for epoch in range(num_epochs):
        mlflow.log_metrics({"loss": loss, "perplexity": ppl}, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Accelerate Setup
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

### VQ-VAE Interface
```python
class VQVAEWrapper:
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        """Encode images to discrete token indices [B, H', W']"""
        pass
    
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode token indices to images [B, C, H, W]"""
        pass
    
    @property
    def vocab_size(self) -> int:
        """Codebook size"""
        pass
```

## Datasets

### Flickr30k (Primary)
- **Location**: `/home/doshlom4/work/final_project/stable_diffusion/notebooks-old/dataset_cache/nlphuji___flickr30k/`
- **Size**: ~30K images with 5 captions each
- **Usage**: Training and validation

### Data Loading Pattern
```python
from datasets import load_dataset

ds = load_dataset(
    "nlphuji/flickr30k",
    cache_dir="./dataset_cache",
    split="test"  # or "train"
)
```

## Dependencies

Key packages:
- `torch>=2.0.0` (CUDA 11.8)
- `accelerate>=0.25.0`
- `transformers>=4.36.0`
- `mlflow>=2.9.0`
- `datasets>=2.15.0`
- `einops>=0.7.0`

## Development Workflow

1. **Prototype** in notebooks (single GPU)
2. **Convert** to Python scripts for training
3. **Submit** via Slurm for multi-GPU
4. **Track** experiments in MLflow
5. **Compare** runs across experiments
