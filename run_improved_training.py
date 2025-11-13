#!/usr/bin/env python
"""
Run improved training to achieve 97% accuracy
"""
import sys
from pathlib import Path

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent / 'model'))

from train_improved import train_improved_model
import torch

if __name__ == '__main__':
    print("="*70)
    print("DR.A-EYE Improved Training - Target: 97% Accuracy")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: Training on CPU (will be slower)")
    
    # Training parameters
    base_path = 'data/raw/DiabeticRetinopathyDataset'
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.0001
    patience = 10
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Early Stopping Patience: {patience}")
    print("\nAdvanced Techniques Enabled:")
    print("  ✓ Focal Loss (handles class imbalance)")
    print("  ✓ Mixup Augmentation (regularization)")
    print("  ✓ Weighted Random Sampling (balanced training)")
    print("  ✓ Advanced Data Augmentation")
    print("  ✓ OneCycleLR Scheduler")
    print("  ✓ Gradient Clipping")
    print("\n" + "="*70)
    print("Starting improved training...")
    print("="*70 + "\n")
    
    # Train the model
    model, history = train_improved_model(
        base_path=base_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        save_dir='model',
        use_focal_loss=True,
        use_mixup=True,
        use_weighted_sampler=True
    )
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)

