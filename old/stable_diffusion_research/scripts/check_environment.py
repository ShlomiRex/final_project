#!/usr/bin/env python3
"""
Environment Checker Script

This script checks for the presence of required libraries, their versions, and potential conflicts
before starting the training process.
"""

import sys
import subprocess

# Required libraries and their versions
REQUIRED_LIBRARIES = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
}

# Helper function to check library versions
def check_library(library, required_version):
    try:
        # Import the library
        lib = __import__(library)
        installed_version = lib.__version__
        if installed_version != required_version:
            print(f"[WARNING] {library} version mismatch: Installed={installed_version}, Required={required_version}")
        else:
            print(f"[OK] {library} version {installed_version} is correct.")
    except ImportError:
        print(f"[ERROR] {library} is not installed.")

# Check all required libraries
def check_environment():
    print("Checking environment...")
    for library, version in REQUIRED_LIBRARIES.items():
        check_library(library, version)

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] CUDA is not available. Training will run on CPU.")
    except ImportError:
        print("[ERROR] PyTorch is not installed. Cannot check CUDA availability.")

    # Check NVIDIA driver
    try:
        subprocess.run(["nvidia-smi"], check=True)
        print("[OK] NVIDIA driver is installed and working.")
    except FileNotFoundError:
        print("[WARNING] NVIDIA driver is not installed or nvidia-smi is not in PATH.")
    except subprocess.CalledProcessError:
        print("[ERROR] nvidia-smi command failed. Check your NVIDIA driver installation.")

if __name__ == "__main__":
    check_environment()