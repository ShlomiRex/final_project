# Research Paper: Text-Conditioned Diffusion Models on MNIST

This repository contains the LaTeX files documenting experiments training text-conditioned diffusion models on MNIST using CLIP embeddings and classifier-free guidance.

## Source Notebook

**Primary Reference**: `stable_diffusion/notebooks-old/complete_new_model/diffusers/train3_working_with_prompts_mnist.ipynb`

This notebook contains:
- Text-to-image diffusion model training on MNIST
- CLIP-conditioned UNet with cross-attention
- Classifier-free guidance implementation
- Ablation study on guidance scales (w = 0, 5, 10, 20, 50, 100)

## Paper Contents

### Current Status
The paper now includes detailed documentation of:

**Methodology** ([methodology.tex](src/methodology.tex)):
- Custom UNet2DConditionModel architecture (2.6M parameters)
- CLIP ViT-B/32 text encoder configuration
- DDPM training procedure with squared cosine schedule
- Classifier-free guidance (CFG) implementation with mathematical formulation

**Experiments** ([experiments.tex](src/experiments.tex)):
- HPC environment setup (PyTorch 2.7.1, CUDA 11.8)
- MNIST dataset with automatic caption generation
- Baseline training: 5 epochs, batch size 512, learning rate 1e-3
- Guidance scale ablation: systematic evaluation of w ∈ {0, 5, 10, 20, 50, 100}

**Results** ([results.tex](src/results.tex)):
- Training convergence analysis
- Qualitative guidance scale comparison
- Visual generation quality assessment
- Optimal guidance scale recommendations (w ∈ [8, 20])

### To Complete
- **abstract.tex**: Summarize key contributions
- **introduction.tex**: Background on diffusion models and CLIP
- **related_work.tex**: Cite DDPM, Stable Diffusion, CLIP, CFG papers
- **discussion.tex**: Interpret findings and limitations
- **conclusion.tex**: Future work and improvements
- **references.bib**: Add BibTeX citations

## Project Structure

- **src/**: Contains all the LaTeX source files for the research paper.
  - **main.tex**: The main LaTeX file that compiles all sections of the paper.
  - **abstract.tex**: Summarizes the key findings and contributions of the research.
  - **introduction.tex**: Outlines the background and significance of the research.
  - **related_work.tex**: Discusses related work in the field and provides context.
  - **methodology.tex**: ✅ **COMPLETED** - Detailed methodology with CLIP-conditioned diffusion
  - **experiments.tex**: ✅ **COMPLETED** - Experimental setup and ablation studies
  - **results.tex**: ✅ **COMPLETED** - Training and generation results
  - **discussion.tex**: Discusses the implications of the results and interprets findings.
  - **conclusion.tex**: Summarizes the main conclusions and suggests future work.
  - **references.bib**: Contains the bibliography in BibTeX format.

- **figures/**: Directory for storing figures (add guidance scale comparison from notebook)
- **tables/**: Directory for storing tables
- **appendix/**: Contains supplementary material
  - **supplementary.tex**: Additional data or explanations.
- **build.sh**: Automated PDF compilation script

## Compiling the Paper

### Quick Build (Recommended)
```bash
./build.sh
```

### Manual Build
```bash
cd src
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Current Output
- **PDF**: `src/main.pdf`
- **Pages**: 10
- **Size**: ~207 KB

## LaTeX Environment

The project uses **TinyTeX** (lightweight LaTeX distribution):
- Installed in `~/bin` and `~/.TinyTeX`
- Added to `~/.bashrc` for persistent PATH access
- Required packages already installed: cite, amsmath, amsfonts, graphicx, hyperref

If you need additional LaTeX packages:
```bash
tlmgr install package-name
```

## Next Steps

1. **Export figures from notebook**:
   ```python
   # In the notebook:
   plt.savefig('../research-paper/figures/guidance_scale_comparison.png', dpi=300, bbox_inches='tight')
   ```

2. **Add key references** to `references.bib`:
   - Ho et al. (2020): Denoising Diffusion Probabilistic Models
   - Ho & Salimans (2022): Classifier-Free Diffusion Guidance
   - Radford et al. (2021): Learning Transferable Visual Models From Natural Language Supervision (CLIP)
   - Rombach et al. (2022): High-Resolution Image Synthesis with Latent Diffusion Models

3. **Complete remaining sections** (abstract, introduction, related work, discussion, conclusion)

4. **Download PDF** to local machine:
   ```bash
   scp username@hpc:~/work/final_project/research-paper/src/main.pdf .
   # Or use VS Code file browser
   ```

## Contribution

All content derived from experimental notebook: `train3_working_with_prompts_mnist.ipynb`

## License

This project is licensed under the MIT License. See the LICENSE file for more details.