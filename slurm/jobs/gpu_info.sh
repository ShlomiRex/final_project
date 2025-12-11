#!/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --output=slurm/logs/gpu_info_%j.out
#SBATCH --error=slurm/logs/gpu_info_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

# ==============================================================================
# GPU Information Job
# ==============================================================================
# Comprehensive GPU and system information for a compute node
# ==============================================================================

echo "=============================================="
echo "GPU and System Information"
echo "=============================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "=============================================="
echo ""

# System information
echo "1. System Information"
echo "----------------------------"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "CPUs: $(nproc)"
echo "Total Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo ""

# GPU Information
echo "2. GPU Information (nvidia-smi)"
echo "----------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not available!"
fi
echo ""

# CUDA Information
echo "3. CUDA Toolkit"
echo "----------------------------"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not in PATH. Searching..."
    for cuda_path in /usr/local/cuda* /opt/cuda*; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            echo "Found at: $cuda_path"
            $cuda_path/bin/nvcc --version
            break
        fi
    done
fi
echo ""

# cuDNN Information
echo "4. cuDNN Library"
echo "----------------------------"
if [ -f /usr/local/cuda/include/cudnn.h ]; then
    CUDNN_MAJOR=$(grep "#define CUDNN_MAJOR" /usr/local/cuda/include/cudnn.h | awk '{print $3}')
    CUDNN_MINOR=$(grep "#define CUDNN_MINOR" /usr/local/cuda/include/cudnn.h | awk '{print $3}')
    CUDNN_PATCH=$(grep "#define CUDNN_PATCHLEVEL" /usr/local/cuda/include/cudnn.h | awk '{print $3}')
    echo "cuDNN Version: $CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCH"
elif [ -f /usr/include/cudnn.h ]; then
    CUDNN_MAJOR=$(grep "#define CUDNN_MAJOR" /usr/include/cudnn.h | awk '{print $3}')
    CUDNN_MINOR=$(grep "#define CUDNN_MINOR" /usr/include/cudnn.h | awk '{print $3}')
    CUDNN_PATCH=$(grep "#define CUDNN_PATCHLEVEL" /usr/include/cudnn.h | awk '{print $3}')
    echo "cuDNN Version: $CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCH"
else
    echo "cuDNN not found or not accessible"
fi
echo ""

# Python and ML libraries
echo "5. Python Environment"
echo "----------------------------"
if command -v python &> /dev/null; then
    echo "Python: $(python --version 2>&1)"
    
    echo ""
    echo "Installed ML packages:"
    python -c "
import sys
packages = ['torch', 'tensorflow', 'jax', 'numpy', 'transformers']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {version}')
        
        # Extra info for PyTorch
        if pkg == 'torch':
            import torch
            print(f'    - CUDA available: {torch.cuda.is_available()}')
            if torch.cuda.is_available():
                print(f'    - CUDA version: {torch.version.cuda}')
                print(f'    - cuDNN version: {torch.backends.cudnn.version()}')
                print(f'    - Device count: {torch.cuda.device_count()}')
    except ImportError:
        print(f'  {pkg}: not installed')
" 2>/dev/null || echo "  Could not check Python packages"
else
    echo "Python not found in PATH"
fi
echo ""

# Conda environments
echo "6. Conda Environments"
echo "----------------------------"
if command -v conda &> /dev/null; then
    echo "Conda version: $(conda --version)"
    echo ""
    echo "Available environments:"
    conda env list
else
    echo "Conda not available"
fi
echo ""

# Module system
echo "7. Loaded Modules"
echo "----------------------------"
if command -v module &> /dev/null; then
    module list 2>&1 || echo "No modules loaded"
else
    echo "Module system not available"
fi
echo ""

echo "=============================================="
echo "Information gathering complete!"
echo "=============================================="
