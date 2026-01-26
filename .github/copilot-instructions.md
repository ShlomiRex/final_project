# GitHub Copilot Instructions

## Project-Specific Coding Standards

### Path Management

**ALWAYS use absolute paths in notebooks and scripts. NEVER use relative paths.**

#### For Jupyter Notebooks

At the top of each notebook, define the project root:

```python
from pathlib import Path
import sys
PROJECT_ROOT = Path("/home/doshlom4/work/final_project")
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")
```

Then use `PROJECT_ROOT` for all file operations:

```python
# ✅ CORRECT - Use absolute paths via PROJECT_ROOT
checkpoint_dir = PROJECT_ROOT / "checkpoints"
outputs_dir = PROJECT_ROOT / "outputs"
dataset_path = PROJECT_ROOT / "dataset_cache"

# ❌ WRONG - Never use relative paths
checkpoint_dir = "./checkpoints"
checkpoint_dir = "../checkpoints"
checkpoint_dir = "../../checkpoints"
```

### Dataset and Cache Management (HPC Environment)

**CRITICAL: NEVER download datasets or cache files to `/home/doshlom4/` unless specifically in `/home/doshlom4/work/` subdirectory.**

- Home directory (`/home/doshlom4/`) has limited quota and resources on HPC
- Work directory (`/home/doshlom4/work/`) has proper storage allocation
- **ALWAYS use:** `/home/doshlom4/work/final_project/dataset_cache` for all dataset downloads

#### HuggingFace Datasets

```python
# ✅ CORRECT - Use project dataset_cache
from datasets import load_dataset
wikiart = load_dataset(
    "huggan/wikiart",
    split="train",
    cache_dir=str(PROJECT_ROOT / "dataset_cache")
)

# ❌ WRONG - Never let HuggingFace use default cache
wikiart = load_dataset("huggan/wikiart")  # Uses ~/.cache/huggingface

# ❌ WRONG - Never cache outside /work
wikiart = load_dataset("huggan/wikiart", cache_dir="/home/doshlom4/.cache")
```

#### PyTorch/Torchvision Datasets

```python
# ✅ CORRECT - Use project dataset_cache
from torchvision.datasets import MNIST, CIFAR10
dataset = MNIST(root=str(PROJECT_ROOT / "dataset_cache"), download=True)
dataset = CIFAR10(root=str(PROJECT_ROOT / "dataset_cache"), download=True)

# ❌ WRONG - Never use home directory
dataset = MNIST(root="~/datasets")
```

#### For Python Scripts

Use absolute paths or calculate the project root dynamically:

```python
from pathlib import Path

# Calculate project root relative to this script
PROJECT_ROOT = Path(__file__).parent.parent  # Adjust as needed
# Or use absolute path
PROJECT_ROOT = Path("/home/doshlom4/work/final_project")

# Then use PROJECT_ROOT for all paths
checkpoint_dir = PROJECT_ROOT / "checkpoints"
```

#### Module Imports

When importing custom modules from the project root:

```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Now you can import project modules
from mnist_classifier import MNISTClassifier
from metrics_mnist_fid import compute_mnist_fid
```

### Directory Structure

Standard project layout:
```
/home/doshlom4/work/final_project/
├── notebooks/
│   └── 01_mnist_experiment/
│       ├── train1_t2i_mnist_cfg.ipynb
│       ├── train2_train_mnist_classifier.ipynb
│       ├── inference1_t2i_mnist_cfg.ipynb
│       └── metrics2_evaluate_t2i_mnist.ipynb
├── checkpoints/          # Model checkpoints
├── outputs/              # Evaluation results, plots
├── dataset_cache/        # Downloaded datasets
├── mnist_classifier.py   # Utility modules
├── metrics_mnist_fid.py
└── .github/
    └── copilot-instructions.md
```

### Why Absolute Paths?

1. **Portability**: Notebooks can be run from any working directory
2. **Clarity**: Always clear what file is being accessed
3. **Maintainability**: Easy to reorganize notebooks without breaking paths
4. **Reproducibility**: No ambiguity about file locations

### Examples

```python
# ✅ Loading a checkpoint (CORRECT)
checkpoint_path = PROJECT_ROOT / "checkpoints" / "mnist_classifier.pt"
model.load_state_dict(torch.load(str(checkpoint_path)))

# ✅ Saving outputs (CORRECT)
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_dir / "results.png"))

# ✅ Loading dataset (CORRECT)
dataset = MNIST(root=str(PROJECT_ROOT / "dataset_cache"), download=True)

# ❌ Using relative paths (WRONG)
checkpoint_path = "../checkpoints/mnist_classifier.pt"  # Don't do this
output_dir = "./outputs"  # Don't do this
dataset = MNIST(root="~/datasets")  # Don't do this
```

### Exception: User Home Directory

The only acceptable use of `~` is when accessing system-wide user directories:

```python
# ✅ OK for system paths
dataset = MNIST(root=os.path.expanduser("~/datasets"))

# But prefer PROJECT_ROOT for project-specific data
dataset = MNIST(root=str(PROJECT_ROOT / "dataset_cache"))
```

---

**Remember: When in doubt, use `PROJECT_ROOT / "path" / "to" / "file"`**

## Jupyter Kernel Configuration

### Default Kernel for This Project

**Kernel Name:** `Python 3.10 (Stable Diffusion - CUDA 11.8)`

**Kernel Location:** `/home/doshlom4/.local/share/jupyter/kernels/stable_diffusion_cuda118/`

**Python Environment:** `/home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025/bin/python`

**Configuration:**
- Python 3.10
- CUDA 11.8 support
- Configured for stable diffusion workloads
- Uses ipykernel for Jupyter integration

### Usage

All notebooks in this project should use the "Python 3.10 (Stable Diffusion - CUDA 11.8)" kernel. This ensures:
- Consistent Python version across all experiments
- CUDA support for GPU acceleration
- All required packages are available (diffusers, transformers, torch, etc.)

To verify the kernel in a notebook, check the top-right corner of VS Code or run:
```python
import sys
print(sys.executable)
# Should output: /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025/bin/python
```
