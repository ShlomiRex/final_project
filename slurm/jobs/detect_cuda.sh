#!/bin/bash
#SBATCH --job-name=detect_cuda
#SBATCH --output=slurm/logs/detect_cuda_%j.out
#SBATCH --error=slurm/logs/detect_cuda_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

# ==============================================================================
# CUDA Detection Job
# ==============================================================================
# Detects CUDA toolkit version and GPU driver version on compute nodes
# ==============================================================================

set -e

echo "=============================================="
echo "CUDA and GPU Detection"
echo "=============================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="
echo ""

# Check if CUDA is available
echo "1. NVIDIA Driver Information"
echo "----------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    
    # Get driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "GPU Driver Version: $DRIVER_VERSION"
else
    echo "ERROR: nvidia-smi not found!"
    echo "This node may not have NVIDIA GPUs or drivers installed."
fi
echo ""

# Check CUDA toolkit version
echo "2. CUDA Toolkit Information"
echo "----------------------------"
if command -v nvcc &> /dev/null; then
    nvcc --version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo ""
    echo "CUDA Toolkit Version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found!"
    echo "CUDA toolkit may not be in PATH."
    echo ""
    
    # Try to find CUDA in common locations
    echo "Searching for CUDA in common locations..."
    for cuda_path in /usr/local/cuda* /opt/cuda* /usr/cuda*; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            echo "Found CUDA at: $cuda_path"
            $cuda_path/bin/nvcc --version
            CUDA_VERSION=$($cuda_path/bin/nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
            echo "CUDA Toolkit Version: $CUDA_VERSION"
        fi
    done
fi
echo ""

# Check CUDA environment variables
echo "3. CUDA Environment Variables"
echo "----------------------------"
echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "CUDA_PATH: ${CUDA_PATH:-Not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo ""

# Check available CUDA modules (if using module system)
echo "4. Available CUDA Modules"
echo "----------------------------"
if command -v module &> /dev/null; then
    module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found"
else
    echo "Module system not available"
fi
echo ""

# GPU details
echo "5. GPU Hardware Details"
echo "----------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Model:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
    
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader
    echo ""
    
    echo "GPU Count:"
    nvidia-smi --query-gpu=count --format=csv,noheader | head -1
    echo ""
    
    echo "Compute Capability:"
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "Not available via nvidia-smi"
fi
echo ""

# Recommendations
echo "=============================================="
echo "RECOMMENDATIONS FOR CONDA ENVIRONMENT"
echo "=============================================="
echo ""

if [ -n "$DRIVER_VERSION" ]; then
    echo "Based on GPU Driver: $DRIVER_VERSION"
    echo ""
    
    # Extract major version
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
    
    # Recommend CUDA toolkit based on driver
    # See: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
    if [ "$DRIVER_MAJOR" -ge 525 ]; then
        echo "✓ Your driver supports CUDA 12.x"
        echo "  Recommended PyTorch installation:"
        echo "    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
        echo "  Or CUDA 11.8:"
        echo "    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    elif [ "$DRIVER_MAJOR" -ge 450 ]; then
        echo "✓ Your driver supports CUDA 11.x"
        echo "  Recommended PyTorch installation:"
        echo "    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    elif [ "$DRIVER_MAJOR" -ge 418 ]; then
        echo "✓ Your driver supports CUDA 10.x"
        echo "  Recommended PyTorch installation:"
        echo "    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch"
    else
        echo "⚠ Old driver version detected"
        echo "  You may need to use CPU-only PyTorch or request a newer node"
    fi
else
    echo "⚠ Could not detect driver version"
    echo "  Check the output above for errors"
fi

echo ""
echo "=============================================="
echo "Detection complete!"
echo "=============================================="
