"""
MNIST-FID and classifier-based evaluation metrics.

This module provides:
1. MNIST-FID computation using MNIST classifier features
2. Real-vs-real baseline computation
3. Conditional accuracy and prompt consistency metrics
4. Visualization utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import os


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution (d,)
        sigma1: Covariance of first distribution (d, d)
        mu2: Mean of second distribution (d,)
        sigma2: Covariance of second distribution (d, d)
        eps: Small value for numerical stability
    
    Returns:
        fid: Fréchet distance (scalar)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


def extract_features_batch(model, images, device, batch_size=128):
    """
    Extract embeddings from MNIST classifier for a batch of images.
    
    Args:
        model: MNISTClassifier with get_embedding method
        images: Tensor of images (N, 1, H, W), should be normalized same as training
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        features: Tensor of embeddings (N, embedding_dim)
    """
    model.eval()
    features_list = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)
            
            batch_features = model.get_embedding(batch)
            features_list.append(batch_features.cpu())
    
    features = torch.cat(features_list, dim=0)
    return features


def compute_statistics(features):
    """
    Compute mean and covariance of features.
    
    Args:
        features: Tensor of shape (N, d)
    
    Returns:
        mu: Mean vector (d,)
        sigma: Covariance matrix (d, d)
    """
    features = features.cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_mnist_fid(
    model,
    real_images,
    fake_images,
    device,
    batch_size=128
):
    """
    Compute MNIST-FID between real and generated images.
    
    Args:
        model: MNISTClassifier for feature extraction
        real_images: Tensor of real images (N, 1, 28, 28)
        fake_images: Tensor of fake images (M, 1, 28, 28)
        device: torch device
        batch_size: Batch size for feature extraction
    
    Returns:
        fid: MNIST-FID score (lower is better)
    """
    # Extract features
    print("Extracting features from real images...")
    real_features = extract_features_batch(model, real_images, device, batch_size)
    
    print("Extracting features from generated images...")
    fake_features = extract_features_batch(model, fake_images, device, batch_size)
    
    # Compute statistics
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    
    # Compute FID
    fid = compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid


def compute_conditional_accuracy(
    model,
    images,
    target_digits,
    device,
    batch_size=128
):
    """
    Compute conditional accuracy: does generated image match prompt digit?
    
    Args:
        model: MNISTClassifier
        images: Tensor of images (N, 1, 28, 28)
        target_digits: List or tensor of target digits (N,)
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        results: Dict with:
            - 'accuracy': Overall accuracy
            - 'per_digit_accuracy': Accuracy per digit class
            - 'predictions': All predictions
            - 'target_probs': Probabilities for target class
            - 'confidences': Max probabilities
    """
    model.eval()
    
    if isinstance(target_digits, list):
        target_digits = torch.tensor(target_digits)
    
    all_predictions = []
    all_target_probs = []
    all_confidences = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing accuracy"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            
            batch_images = images[start_idx:end_idx].to(device)
            batch_targets = target_digits[start_idx:end_idx]
            
            logits = model(batch_images)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            confidences, predictions = probs.max(dim=1)
            
            # Get target probabilities
            target_probs = probs[torch.arange(len(batch_targets)), batch_targets.long()]
            
            all_predictions.append(predictions.cpu())
            all_target_probs.append(target_probs.cpu())
            all_confidences.append(confidences.cpu())
    
    predictions = torch.cat(all_predictions)
    target_probs = torch.cat(all_target_probs)
    confidences = torch.cat(all_confidences)
    
    # Compute overall accuracy
    correct = (predictions == target_digits).float()
    accuracy = correct.mean().item() * 100
    
    # Per-digit accuracy
    per_digit_accuracy = {}
    for digit in range(10):
        mask = target_digits == digit
        if mask.sum() > 0:
            digit_acc = correct[mask].mean().item() * 100
            per_digit_accuracy[digit] = digit_acc
    
    results = {
        'accuracy': accuracy,
        'per_digit_accuracy': per_digit_accuracy,
        'predictions': predictions,
        'target_probs': target_probs,
        'confidences': confidences,
        'targets': target_digits
    }
    
    return results


def visualize_conditional_accuracy(
    images,
    targets,
    predictions,
    confidences,
    num_samples=20,
    save_path=None
):
    """
    Visualize generated images with predictions and targets.
    
    Args:
        images: Tensor of images (N, 1, 28, 28)
        targets: Target digits (N,)
        predictions: Predicted digits (N,)
        confidences: Prediction confidences (N,)
        num_samples: Number of samples to display
        save_path: Optional path to save figure
    """
    num_samples = min(num_samples, len(images))
    
    # Sample indices
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < num_samples:
            i = indices[idx]
            img = images[i].squeeze().cpu().numpy()
            target = targets[i].item()
            pred = predictions[i].item()
            conf = confidences[i].item()
            
            ax.imshow(img, cmap='gray')
            
            # Color: green if correct, red if wrong
            color = 'green' if target == pred else 'red'
            ax.set_title(f"Target: {target}, Pred: {pred}\nConf: {conf:.2f}", color=color)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_confusion_matrix(targets, predictions, save_path=None):
    """
    Plot confusion matrix for digit classification.
    
    Args:
        targets: Target digits (N,)
        predictions: Predicted digits (N,)
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Digit')
    plt.ylabel('Target Digit')
    plt.title('Confusion Matrix: Target vs Predicted Digits')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_fid_vs_guidance(
    guidance_scales,
    fid_values,
    baseline_fid=None,
    save_path=None
):
    """
    Plot MNIST-FID vs guidance scale.
    
    Args:
        guidance_scales: List of guidance scales
        fid_values: List of corresponding FID values
        baseline_fid: Real-vs-real baseline FID (optional)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(guidance_scales, fid_values, marker='o', linewidth=2, markersize=8, label='Model FID')
    
    if baseline_fid is not None:
        plt.axhline(baseline_fid, linestyle='--', linewidth=2, color='red', 
                   label=f'Real-vs-Real Baseline ({baseline_fid:.2f})')
    
    plt.xlabel('Guidance Scale (w)', fontsize=12)
    plt.ylabel('MNIST-FID', fontsize=12)
    plt.title('MNIST-FID vs Classifier-Free Guidance Scale', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved FID plot to {save_path}")
    
    plt.show()


def plot_accuracy_vs_guidance(
    guidance_scales,
    accuracies,
    save_path=None
):
    """
    Plot conditional accuracy vs guidance scale.
    
    Args:
        guidance_scales: List of guidance scales
        accuracies: List of corresponding accuracies
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(guidance_scales, accuracies, marker='o', linewidth=2, markersize=8, color='green')
    
    plt.xlabel('Guidance Scale (w)', fontsize=12)
    plt.ylabel('Prompt Accuracy (%)', fontsize=12)
    plt.title('Prompt Consistency vs Classifier-Free Guidance Scale', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to {save_path}")
    
    plt.show()


def generate_evaluation_report(
    fid_results: Dict,
    accuracy_results: Dict,
    baseline_fid: float,
    num_samples: int,
    save_path: Optional[str] = None
):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        fid_results: Dict mapping guidance_scale -> FID value
        accuracy_results: Dict mapping guidance_scale -> accuracy results dict
        baseline_fid: Real-vs-real baseline FID
        num_samples: Number of samples used
        save_path: Optional path to save report as text file
    """
    report = []
    report.append("=" * 70)
    report.append("MNIST Text-to-Image Diffusion Model Evaluation Report")
    report.append("=" * 70)
    report.append(f"Number of samples: {num_samples}")
    report.append(f"Real-vs-Real Baseline FID: {baseline_fid:.2f}")
    report.append("")
    
    report.append("-" * 70)
    report.append("MNIST-FID Results (lower is better)")
    report.append("-" * 70)
    for w in sorted(fid_results.keys()):
        fid_val = fid_results[w]
        report.append(f"Guidance Scale w={w:>3}: FID = {fid_val:7.2f}")
    report.append("")
    
    report.append("-" * 70)
    report.append("Conditional Accuracy Results (higher is better)")
    report.append("-" * 70)
    for w in sorted(accuracy_results.keys()):
        acc_data = accuracy_results[w]
        acc = acc_data['accuracy']
        report.append(f"Guidance Scale w={w:>3}: Accuracy = {acc:6.2f}%")
        
        # Per-digit breakdown
        report.append(f"  Per-digit accuracy:")
        for digit in range(10):
            if digit in acc_data['per_digit_accuracy']:
                digit_acc = acc_data['per_digit_accuracy'][digit]
                report.append(f"    Digit {digit}: {digit_acc:6.2f}%")
        report.append("")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {save_path}")
    
    return report_text


if __name__ == "__main__":
    print("This module provides MNIST-FID and evaluation metrics.")
    print("Import and use in your notebook or script.")
