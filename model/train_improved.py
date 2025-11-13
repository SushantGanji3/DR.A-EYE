"""
Improved training script with advanced techniques for 97% accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from model import DiabeticRetinopathyModel
from dataset import get_dataloaders, prepare_data_splits
from improve_accuracy import (
    get_class_weights, create_weighted_sampler, FocalLoss,
    get_advanced_transforms, get_advanced_optimizer, mixup_data, mixup_criterion
)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_acc = 0.0
        
    def __call__(self, val_loss, val_acc, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta or val_acc > self.best_acc:
            if val_acc > self.best_acc:
                self.best_acc = val_acc
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


def train_epoch_improved(model, train_loader, criterion, optimizer, device, use_mixup=True, class_weights=None):
    """Improved training with mixup and weighted loss"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def train_improved_model(
    base_path, 
    num_epochs=100, 
    batch_size=32, 
    learning_rate=0.0001,
    patience=10, 
    device=None, 
    save_dir='../model',
    use_focal_loss=True,
    use_mixup=True,
    use_weighted_sampler=True
):
    """
    Improved training function with advanced techniques
    
    Args:
        base_path: Path to dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate (lower for better convergence)
        patience: Early stopping patience
        device: Device to train on
        save_dir: Directory to save model
        use_focal_loss: Use Focal Loss for imbalanced data
        use_mixup: Use Mixup augmentation
        use_weighted_sampler: Use weighted sampling for class balance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data splits
    print("Loading and preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_data_splits(base_path, test_size=0.15, val_size=0.15, random_state=42)
    
    # Compute class weights
    class_weights = get_class_weights(train_labels)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Create datasets with advanced transforms
    from dataset import DiabeticRetinopathyDataset
    train_dataset = DiabeticRetinopathyDataset(
        train_paths, train_labels, transform=get_advanced_transforms(train=True)
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_paths, val_labels, transform=get_advanced_transforms(train=False)
    )
    test_dataset = DiabeticRetinopathyDataset(
        test_paths, test_labels, transform=get_advanced_transforms(train=False)
    )
    
    # Create weighted sampler if needed
    from torch.utils.data import DataLoader
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model
    print("Initializing model...")
    model = DiabeticRetinopathyModel(num_classes=5, pretrained=True)
    model = model.to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print("Using Focal Loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropy Loss")
    
    # Optimizer
    optimizer = get_advanced_optimizer(model, learning_rate=learning_rate)
    
    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,  # Peak learning rate
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    print("\n" + "="*60)
    print("Starting Improved Training...")
    print("="*60)
    print(f"Techniques enabled:")
    print(f"  - Focal Loss: {use_focal_loss}")
    print(f"  - Mixup Augmentation: {use_mixup}")
    print(f"  - Weighted Sampler: {use_weighted_sampler}")
    print(f"  - Advanced Augmentation: Yes")
    print(f"  - Gradient Clipping: Yes")
    print(f"  - OneCycleLR Scheduler: Yes")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch_improved(
            model, train_loader, criterion, optimizer, device, 
            use_mixup=use_mixup, class_weights=class_weights
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'best_resnet18_improved.pth')
            print(f"✓ Saved best model with val_acc: {best_val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, val_acc, model):
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_dir / 'best_resnet18_improved.pth'))
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Get detailed test metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Calculate misdiagnosis rate
    misdiagnosis_rate = 1 - test_acc
    print(f"\nMisdiagnosis Rate: {misdiagnosis_rate*100:.2f}%")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Improved Model)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix_improved.png', dpi=150)
    plt.close()
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy improvement over epochs
    improvement = [acc - history['val_acc'][0] for acc in history['val_acc']]
    axes[1, 1].plot(improvement, linewidth=2, color='green')
    axes[1, 1].set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy Gain', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history_improved.png', dpi=150)
    plt.close()
    
    # Save history to JSON
    history['test_acc'] = float(test_acc)
    history['test_loss'] = float(test_loss)
    history['misdiagnosis_rate'] = float(misdiagnosis_rate)
    history['best_val_acc'] = float(best_val_acc)
    
    with open(save_dir / 'training_history_improved.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Improvement: {test_acc - 0.82:.4f} (+{(test_acc - 0.82)*100:.2f}%)")
    print(f"Model saved to: {save_dir / 'best_resnet18_improved.pth'}")
    
    return model, history


if __name__ == '__main__':
    base_path = '../data/raw/DiabeticRetinopathyDataset'
    train_improved_model(
        base_path=base_path,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.0001,
        patience=10,
        use_focal_loss=True,
        use_mixup=True,
        use_weighted_sampler=True
    )

