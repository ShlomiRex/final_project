"""
MNIST Classifier for computing MNIST-specific FID and conditional accuracy.

This module provides:
1. A LeNet-style CNN classifier for MNIST
2. Training and evaluation utilities
3. Feature extraction for MNIST-FID computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm
import os


class MNISTClassifier(nn.Module):
    """
    LeNet-style CNN for MNIST classification.
    
    Architecture:
    - Conv1: 1 -> 32 channels, 5x5 kernel
    - Conv2: 32 -> 64 channels, 5x5 kernel
    - FC1: 1024 -> 128 (embedding layer)
    - FC2: 128 -> 10 (logits)
    """
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 1, 28, 28)
            return_embedding: If True, return (logits, embeddings), else just logits
        
        Returns:
            If return_embedding=False: logits (N, 10)
            If return_embedding=True: (logits, embeddings) where embeddings is (N, embedding_dim)
        """
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC1 -> embedding layer
        embedding = F.relu(self.fc1(x))
        
        # Dropout and final layer
        x = self.dropout(embedding)
        logits = self.fc2(x)
        
        if return_embedding:
            return logits, embedding
        return logits
    
    def get_embedding(self, x):
        """
        Extract embeddings (penultimate layer features) for FID computation.
        
        Args:
            x: Input tensor of shape (N, 1, 28, 28), normalized to [-1, 1] or [0, 1]
        
        Returns:
            Embeddings of shape (N, embedding_dim)
        """
        with torch.no_grad():
            _, embedding = self.forward(x, return_embedding=True)
        return embedding


def train_mnist_classifier(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs=10,
    lr=0.001,
    save_path=None
):
    """
    Train the MNIST classifier.
    
    Args:
        model: MNISTClassifier instance
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        device: torch device
        num_epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the best model (optional)
    
    Returns:
        model: Trained model
        history: Dict with 'train_loss', 'train_acc', 'test_acc' lists
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*train_correct/train_total:.2f}%"})
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Testing
        test_acc = evaluate_mnist_classifier(model, test_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if save_path and test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'embedding_dim': model.embedding_dim
            }, save_path)
            print(f"Saved best model with test accuracy: {test_acc:.2f}%")
    
    return model, history


def evaluate_mnist_classifier(model, test_loader, device):
    """
    Evaluate the classifier on test set.
    
    Args:
        model: MNISTClassifier instance
        test_loader: DataLoader for test set
        device: torch device
    
    Returns:
        test_acc: Test accuracy (percentage)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    return test_acc


def load_mnist_classifier(checkpoint_path, device, embedding_dim=128):
    """
    Load a trained MNIST classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        embedding_dim: Embedding dimension (default 128)
    
    Returns:
        model: Loaded MNISTClassifier
        checkpoint: Full checkpoint dict (with metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get embedding_dim from checkpoint if available
    if 'embedding_dim' in checkpoint:
        embedding_dim = checkpoint['embedding_dim']
    
    model = MNISTClassifier(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def get_mnist_dataloaders(batch_size=128, normalize_range='neg1to1'):
    """
    Get MNIST train and test dataloaders with appropriate normalization.
    
    Args:
        batch_size: Batch size for dataloaders
        normalize_range: 'neg1to1' for [-1, 1] or '0to1' for [0, 1]
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    if normalize_range == 'neg1to1':
        # Normalize to [-1, 1] (mean=0.1307, std=0.3081 are MNIST stats)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif normalize_range == '0to1':
        # Just convert to tensor (values in [0, 1])
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unknown normalize_range: {normalize_range}")
    
    train_dataset = MNIST(root='./dataset_cache', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./dataset_cache', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    # Quick training script for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, _, _ = get_mnist_dataloaders(batch_size=128)
    
    # Create model
    model = MNISTClassifier(embedding_dim=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "mnist_classifier.pt")
    
    model, history = train_mnist_classifier(
        model, train_loader, test_loader, device,
        num_epochs=10, lr=0.001, save_path=save_path
    )
    
    # Load best model and evaluate
    model, checkpoint = load_mnist_classifier(save_path, device)
    test_acc = evaluate_mnist_classifier(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
