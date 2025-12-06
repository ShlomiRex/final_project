# Final Project

Title: Text-Conditioned Image Generation with Stable Diffusion

Table of Contents:

- [Final Project](#final-project)
  - [Running Jupyter on HPC Cluster](#running-jupyter-on-hpc-cluster)
  - [Projects and resources that helped me along the way](#projects-and-resources-that-helped-me-along-the-way)
  - [Virtual Environment Setup (pytorch CPU)](#virtual-environment-setup-pytorch-cpu)
  - [Resources \& Tutorials used](#resources--tutorials-used)
  - [Paper used](#paper-used)

## Running Jupyter on HPC Cluster

**Quick Start - Automated Method (Recommended):**

### Step 1: Check GPU Availability

Before launching Jupyter, check which nodes have available resources:

```bash
bash slurm/check_gpu_availability.sh
```

This shows:
- Which GPU nodes have available GPUs (Titan vs A100)
- CPU-only nodes for development
- Current cluster status and recommendations

### Step 2: Using the Launcher Script

On HPC (SSH to login node), run:

```bash
# For quick access (no GPU - starts faster, good for development)
bash slurm/start_jupyter.sh nogpu

# OR for GPU access (may take longer to start)
bash slurm/start_jupyter.sh

# OR request a specific GPU node
bash slurm/start_jupyter.sh gpu8    # 8x A100 GPUs
bash slurm/start_jupyter.sh gpu7    # 8x A100 GPUs + 2TB RAM (best for large models)
bash slurm/start_jupyter.sh gpu1    # 2x Titan GPUs (usually faster to get)

# OR request a specific CPU node
bash slurm/start_jupyter.sh cn43    # 80 CPUs, often available
```

**⚡ Tip:** Use `nogpu` option for faster startup when you don't need GPU for development/testing. The cluster often has limited GPU availability.

**📊 Node Selection Guide:** See [slurm/NODE_SELECTION_GUIDE.md](slurm/NODE_SELECTION_GUIDE.md) for detailed comparison of nodes and when to use each.

This script will:
1. Find an available port automatically
2. Submit the Jupyter job
3. Wait for it to start
4. Print the exact SSH tunnel command for you to run on your local machine

**Example output:**
```
====================================================================
STEP 2: Run this command on your LOCAL MACHINE (Windows/Mac/Linux):
====================================================================

ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9123:gpu8.hpc.pub.lan:9123 DOSHLOM4@login8.openu.ac.il

====================================================================
STEP 3: Open your browser
====================================================================

Go to: http://localhost:9123
```

Simply copy the SSH command and run it on your Windows machine, then open the browser URL.

**When finished:**
- Press `Ctrl+C` in the SSH tunnel terminal
- The script will tell you the `scancel` command to stop the job

---

**Manual Method (Advanced):**

### 1. On HPC (SSH to login node):

```bash
# Submit the Jupyter job
sbatch slurm/jupyter_notebook_interactive.sh

# Check job status (wait until state shows 'R' for Running)
squeue -u $USER

# Get connection details (replace 123456 with your actual job ID)
bash slurm/connect_jupyter.sh 123456
```

The output will show you the `<COMPUTE_NODE>` (e.g., `gpu8.hpc.pub.lan`) and `<JUPYTER_PORT>` (e.g., `8889`).

### 2. On Your Local Machine:

Create an SSH tunnel using the details from step 1:

```bash
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L <LOCAL_PORT>:<COMPUTE_NODE>:<JUPYTER_PORT> DOSHLOM4@login8.openu.ac.il
```

**Example with actual values:**
```bash
ssh -J DOSHLOM4@sheshet.cslab.openu.ac.il -N -L 9999:gpu8.hpc.pub.lan:8889 DOSHLOM4@login8.openu.ac.il
```

- `<LOCAL_PORT>`: Choose any available port on your machine (e.g., `9999`, `8000`)
- `<COMPUTE_NODE>`: From job output (e.g., `gpu8.hpc.pub.lan`)
- `<JUPYTER_PORT>`: From job output (e.g., `8889`)

**Note:** The SSH command will appear to hang - this is normal. Keep it running.

### 3. Open Browser:

Go to `http://localhost:<LOCAL_PORT>` (e.g., `http://localhost:9999`)

**When finished:**
- Press `Ctrl+C` in the SSH tunnel terminal
- Cancel the job: `scancel <JOB_ID>`

**For detailed documentation:** See [slurm/JUPYTER_SETUP.md](slurm/JUPYTER_SETUP.md)

---

## Projects and resources that helped me along the way

- **U-Net**: I created simple U-Net (from the paper) to segment images of cats and dogs on the Oxford IIIT Pets dataset. Github: https://github.com/ShlomiRex/image-segmentation
- **Variational Autoencoder**: Trained simple VAE on MNIST digit dataset to classify images of handwritten digits, as well as to denoise such images.
  - BlendDigits: online demo that uses VAE to blend two digits together. It demonstrates the continuous latent space of the VAE.
    - Demo: https://blenddigits.shlomidom.com
    - Github: https://github.com/ShlomiRex/BlendDigits
  - Interactive digit denoiser where the user changes mean, variance of the noise and the model denoises the image.
    - Github: https://github.com/ShlomiRex/interactive_denoiser
  - Interactive digit classifier where the user draws a digit and the model classifies it.
    - Github: https://github.com/ShlomiRex/interactive_digit_classifier
- **Noise schedulers**: linear and cosine noise schedulers appear in this project under `src/experiments/noise_schedulers`.

## Virtual Environment Setup (pytorch CPU)

We have to make sure we are using the same python version and packages as my development environment in order not to break my project.

Install `virtualenv`, `pyenv`, `poetry`. Then run (this installs pytorch in CPU mode):

```
pyenv local 3.11.8
poetry env use 3.11.8
poetry install
poetry env activate

# Add CUDA support after activating the environment (read next section)
poe install_cuda
```

Make sure that the command `poetry env activate` run again in your terminal to make sure the active environment is running (example):

`source /Users/user/Library/Caches/pypoetry/virtualenvs/text-conditioned-image-generation-using-st-l7k_OWS4-py3.11/bin/activate` 

(or on Windows (for example): `C:\Users\Shlomi\AppData\Local\pypoetry\Cache\virtualenvs\text-conditioned-image-generation-using-st-35DVCAXA-py3.11\Scripts\activate.ps1`)

We can be sure we created a new environment by running `pip list`, we should see only pip in the list. If not, we can remove the environment folder (given by `poetry env info`).

## Resources & Tutorials used

- Understanding Score Matching, Langevin Dynamics Sampling, Stochastic Differential Equations, the math behind diffusion and basically how diffusion started: https://www.youtube.com/watch?v=B4oHJpEJBAA

## Paper used

- DDPM paper: https://arxiv.org/abs/2006.11239
- OpenAI paper about using cosine noise scheduler: https://arxiv.org/abs/2102.09672
- OpenAI paper improving the architecture of the U-Net by adding normalization layers, residual connections, and attention layers: https://arxiv.org/abs/2105.05233
- Cool website that shows math of diffusion models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
