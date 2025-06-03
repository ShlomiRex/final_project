# Final Project

Title: Text-Conditioned Image Generation with Stable Diffusion

Table of Contents:

- [Final Project](#final-project)
  - [Projects and resources that helped me along the way](#projects-and-resources-that-helped-me-along-the-way)
  - [Virtual Environment Setup (pytorch CPU)](#virtual-environment-setup-pytorch-cpu)
  - [Resources \& Tutorials used](#resources--tutorials-used)
  - [Paper used](#paper-used)

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
