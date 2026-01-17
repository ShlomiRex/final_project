# MNIST-specific “FID / CLIP-score-like” evaluation plan

## Goal
We want a metric that answers: **“Do the generated images match the MNIST dataset distribution?”** and (optionally) **“Do generated images match the text prompt digit?”**

Using standard **Inception-FID** (ImageNet InceptionV3) often gives confusingly large values on MNIST because:
- MNIST is *not* ImageNet-like; Inception features are a poor semantic embedding for 28×28 digits.
- MNIST must be upsampled to 299×299, amplifying tiny artifacts/background shifts.

So we’ll use MNIST-trained features instead.

---

## What we’ll implement

### 1) MNIST-FID (FID-like score using MNIST classifier features)
Compute the usual Fréchet distance, but **in the feature space of a classifier trained on MNIST**, e.g. a small CNN/LeNet.

**Definition (same as FID):**
- Let $f(x) \in \mathbb{R}^d$ be an embedding from a chosen layer of the MNIST classifier (e.g., penultimate layer).
- For real images $x_r$ and generated images $x_g$:
  - $\mu_r = \mathbb{E}[f(x_r)]$, $\Sigma_r = \mathrm{Cov}(f(x_r))$
  - $\mu_g = \mathbb{E}[f(x_g)]$, $\Sigma_g = \mathrm{Cov}(f(x_g))$

Then
$$
\mathrm{MNIST\text{-}FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}).
$$

**Interpretation:** lower is better; it measures distribution similarity *in a digit-relevant feature space*.

**Important baselines:**
- **Real-vs-real baseline**: compute MNIST-FID between two random real subsets. This is your “noise floor.”
- Model-vs-real should ideally be close to (or not wildly above) real-vs-real.

### 2) Classifier-based “score” (CLIP-score analogue)
CLIP-score is “image-text similarity in CLIP embedding.” For MNIST, we can define a simpler and more meaningful analogue using the MNIST classifier:

#### 2a) Conditional accuracy (prompt consistency)
If prompts are like: `"A handwritten digit 7"`, then for a generated image $x$:
- Let classifier output probabilities $p(y|x)$ for digits $y \in \{0..9\}$.
- Parse the target digit $t$ from the prompt.

Metrics:
- **Top-1 prompt accuracy**: $\mathbb{1}[\arg\max_y p(y|x) = t]$
- **Average target probability**: $\mathbb{E}[p(t|x)]$
- **Calibration-ish**: histogram of $p(t|x)$ for correct vs incorrect samples.

This directly answers: **“Does the image match the label in the prompt?”**

#### 2b) “MNIST-CLIP score” (optional)
If you still want a cosine-similarity style score:
- Represent text prompt as a one-hot vector or learned embedding of the digit class.
- Represent image as the classifier penultimate embedding $f(x)$.
- Learn a small linear head that maps $f(x)$ to a digit embedding space.

This is usually unnecessary for MNIST; conditional accuracy + MNIST-FID is simpler and more defensible.

---

## Design choices

### Which feature layer to use
Recommended options:
- **Penultimate layer (embedding)**: best tradeoff (semantic enough, low dimension).
- Avoid using logits directly if possible; embeddings are typically more stable.

### Input normalization
To get consistent results:
- Ensure **real and generated images go through the same preprocessing**.
- If the classifier was trained on normalized MNIST (e.g., mean=0.1307 std=0.3081 or [-1,1]), then apply the same to generated samples before inference.

### Sample size guidance
- For stable covariance estimates, use **>= 1,000** real and **>= 1,000** generated images.
- For quick iteration, use 200–500.

### Reproducibility
- Fix random seeds for sampling.
- Cache generated samples/features so you don’t re-run slow diffusion generation each time.

---

## Implementation plan (for the notebook / repo)

### Step 0 — Decide scope
- Evaluate **unconditional quality** (distribution match) with MNIST-FID.
- Evaluate **text conditioning** with conditional accuracy vs prompt digit.

### Step 1 — Train or load an MNIST classifier
Option A: Train a small CNN in-notebook (fast, reproducible).
- Architecture: LeNet-ish.
- Train for a few epochs until test accuracy ~98–99%.
- Save weights to `checkpoints/mnist_classifier.pt`.

Option B: Ship pretrained weights (fastest inference; requires storing file).

Acceptance criteria:
- Classifier test accuracy >= 98% on MNIST test.

### Step 2 — Add a “feature extractor” wrapper
- Load classifier, set `.eval()`.
- Expose a method returning the chosen embedding layer $f(x)$.

Example (conceptual):
- `features = model.get_embedding(batch)` -> shape `[N, d]`

### Step 3 — Create consistent datasets for evaluation
- Real set: sample `N` images from MNIST test set.
- Generated set: sample `N` images from your diffusion model.
  - For text-conditional generation: sample digits uniformly and generate from prompts.

Store both:
- Optionally save generated images to disk (e.g., `outputs/generated_eval/*.png`) for reuse.

### Step 4 — Compute MNIST-FID
- Compute features for real and generated batches.
- Compute $\mu$ and $\Sigma$ for each.
- Compute Fréchet distance (use `scipy.linalg.sqrtm` or a stable torch implementation).

Add **real-vs-real baseline**:
- Split real set into `real_A`, `real_B`, compute MNIST-FID(real_A, real_B).

Sanity checks:
- MNIST-FID(real_A, real_A) should be close to 0.
- MNIST-FID(real_A, real_B) should be small and stable.

### Step 5 — Compute classifier-based scores
For each generated image + prompt digit target `t`:
- Predicted digit: `argmax p(y|x)`
- Record:
  - prompt accuracy
  - avg target probability $p(t|x)$
  - confusion matrix

Report:
- Accuracy overall
- Accuracy per digit
- Mean/median target probability

### Step 6 — Visualizations
- Grid of generated images with:
  - predicted digit
  - target digit
  - confidence
- Plot:
  - MNIST-FID vs guidance scale
  - Prompt accuracy vs guidance scale

### Step 7 — Performance + caching
- Cache features:
  - Save embeddings to `outputs/mnist_fid_cache/*.pt` with metadata (seed, guidance, N).
- Make evaluation rerunnable without regenerating images.

---

## What to report in the paper / README
Recommended to report both distribution and conditional metrics:
- **MNIST-FID (real-vs-real baseline + model-vs-real)**
- **Prompt consistency accuracy**
- Sample size `N`, seed, classifier accuracy
- Guidance scale used

Suggested phrasing:
- “We compute an MNIST-specific FID using features from an MNIST-trained classifier (LeNet-style). This better reflects digit semantics than ImageNet Inception features.”

---

## Optional upgrades
- **Precision/Recall for generative models** in feature space (helps detect mode collapse).
- **Per-class MNIST-FID**: compute FID conditioned on each digit class.

---

## Risks / pitfalls
- Using too few samples -> covariance unstable -> misleading MNIST-FID.
- Real and generated preprocessing mismatch -> inflated scores.
- If prompts are imbalanced (not uniform digits), conditional accuracy can look misleading.

---

## Concrete deliverables (when implementing)
- `mnist_classifier.py` (LeNet + feature extraction)
- `metrics_mnist_fid.py` (MNIST-FID computation)
- Notebook cell(s):
  - train/load classifier
  - generate evaluation set
  - compute MNIST-FID + baselines
  - compute prompt accuracy + plots
