"""
WikiArt Art Style Classifier for computing classification accuracy on generated images.

This module provides:
1. A ResNet-based CNN classifier for 27 WikiArt art styles
2. Training and evaluation utilities
3. Feature extraction for FID computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import sys

# Import project configuration
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import WIKIART_STYLES, CHECKPOINTS_DIR


class WikiArtClassifier(nn.Module):
    """
    ResNet-18 based classifier for WikiArt art styles.
    
    Architecture:
    - ResNet-18 backbone (pretrained on ImageNet)
    - Custom classification head for 27 art styles
    - Embedding layer (512-dim) for feature extraction
    
    Usage:
        from wikiart_classifier import WikiArtClassifier
        
        classifier = WikiArtClassifier(num_classes=27)
        classifier.load_pretrained()  # Load from checkpoint
        
        # Classification
        logits = classifier(images)
        predictions = logits.argmax(dim=1)
        
        # Feature extraction
        embeddings = classifier.get_embedding(images)
    """
    
    def __init__(self, num_classes: int = 27, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.style_names = WIKIART_STYLES
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        # Replace the final FC layer with embedding + classification layers
        self.backbone.fc = nn.Identity()  # Remove original FC
        
        # Embedding layer
        self.embedding = nn.Linear(num_features, embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, H, W), normalized to [-1, 1] or [0, 1]
            return_embedding: If True, return (logits, embeddings)
        
        Returns:
            If return_embedding=False: logits (N, num_classes)
            If return_embedding=True: (logits, embeddings) where embeddings is (N, embedding_dim)
        """
        # Get backbone features
        features = self.backbone(x)  # (N, 512)
        
        # Embedding
        embedding = self.embedding(features)  # (N, embedding_dim)
        
        # Classification
        logits = self.classifier(embedding)  # (N, num_classes)
        
        if return_embedding:
            return logits, embedding
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for feature-based evaluation (e.g., FID).
        
        Args:
            x: Input tensor of shape (N, 3, H, W)
        
        Returns:
            Embeddings of shape (N, embedding_dim)
        """
        with torch.no_grad():
            _, embedding = self.forward(x, return_embedding=True)
        return embedding
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and confidence scores.
        
        Args:
            x: Input tensor of shape (N, 3, H, W)
        
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predictions = probs.max(dim=1)
        return predictions, confidence
    
    def predict_style_names(self, x: torch.Tensor) -> list[str]:
        """
        Get predicted style names.
        
        Args:
            x: Input tensor of shape (N, 3, H, W)
        
        Returns:
            List of style name strings
        """
        predictions, _ = self.predict(x)
        return [self.style_names[p.item()] for p in predictions]


def get_wikiart_classifier_checkpoint_path() -> Path:
    """Get path to WikiArt classifier checkpoint."""
    return CHECKPOINTS_DIR / "wikiart_classifier.pt"


def load_wikiart_classifier(device: torch.device = None, checkpoint_path: str = None):
    """
    Load WikiArt classifier from checkpoint.
    
    Args:
        device: Device to load the model to
        checkpoint_path: Path to checkpoint (uses default if None)
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_path is None:
        checkpoint_path = get_wikiart_classifier_checkpoint_path()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = WikiArtClassifier(
        num_classes=checkpoint.get('num_classes', 27),
        embedding_dim=checkpoint.get('embedding_dim', 512)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded WikiArt classifier with test accuracy: {checkpoint.get('test_acc', 'N/A')}%")
    
    return model, checkpoint


def train_wikiart_classifier(
    model: WikiArtClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 0.001,
    save_path: Path = None
):
    """
    Train the WikiArt classifier.
    
    Args:
        model: WikiArtClassifier instance
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        device: torch device
        num_epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the best model
    
    Returns:
        model: Trained model
        history: Dict with 'train_loss', 'train_acc', 'test_acc' lists
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
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
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Testing
        test_acc = evaluate_wikiart_classifier(model, test_loader, device)
        
        scheduler.step()
        
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
                'num_classes': model.num_classes,
                'embedding_dim': model.embedding_dim,
            }, save_path)
            print(f"  ✓ Saved best model with test accuracy: {test_acc:.2f}%")
    
    return model, history


def evaluate_wikiart_classifier(
    model: WikiArtClassifier, 
    test_loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Evaluate the classifier on test set.
    
    Args:
        model: WikiArtClassifier instance
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


def compute_per_class_accuracy(
    model: WikiArtClassifier,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Compute accuracy for each art style.
    
    Args:
        model: WikiArtClassifier instance
        test_loader: DataLoader for test set
        device: torch device
    
    Returns:
        Dict mapping style names to accuracy percentages
    """
    model.eval()
    
    # Track correct/total for each class
    class_correct = {i: 0 for i in range(len(WIKIART_STYLES))}
    class_total = {i: 0 for i in range(len(WIKIART_STYLES))}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for label, pred in zip(labels, predicted):
                label = label.item()
                class_total[label] += 1
                if pred.item() == label:
                    class_correct[label] += 1
    
    # Compute per-class accuracy
    per_class_acc = {}
    for class_idx in range(len(WIKIART_STYLES)):
        style_name = WIKIART_STYLES[class_idx]
        if class_total[class_idx] > 0:
            acc = 100. * class_correct[class_idx] / class_total[class_idx]
        else:
            acc = 0.0
        per_class_acc[style_name] = acc
    
    return per_class_acc


class WikiArtDataset(Dataset):
    """
    PyTorch Dataset for WikiArt images from directory structure.
    
    Expected structure:
        base_dir/
            style_0_Abstract_Expressionism/
                image1.png
                image2.png
            style_1_Action_painting/
                ...
    """
    
    def __init__(self, base_dir: Path, transform=None):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        
        # Find all images
        for style_idx, style_name in enumerate(WIKIART_STYLES):
            style_dir = self.base_dir / f"style_{style_idx}_{style_name}"
            if style_dir.exists():
                for img_path in style_dir.glob("*.png"):
                    self.samples.append((img_path, style_idx))
        
        print(f"WikiArtDataset: Found {len(self.samples)} images in {base_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_wikiart_transforms(image_size: int = 128):
    """
    Get standard transforms for WikiArt classifier.
    
    Args:
        image_size: Target image size
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, test_transform


if __name__ == "__main__":
    # Quick test
    print("WikiArt Classifier Module")
    print(f"Number of styles: {len(WIKIART_STYLES)}")
    print(f"Styles: {WIKIART_STYLES[:5]}...")
    
    # Test model creation
    model = WikiArtClassifier(num_classes=27)
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
