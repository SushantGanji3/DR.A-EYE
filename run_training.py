#!/usr/bin/env python
"""
Direct training script for Diabetic Retinopathy model
Run this script to train the model without Jupyter notebook
"""
import sys
from pathlib import Path

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent / 'model'))

from train import train_model
import torch

if __name__ == '__main__':
    print("="*60)
    print("DR.A-EYE Model Training")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: Training on CPU (will be slower)")
    
    # Training parameters
    base_path = 'data/raw/DiabeticRetinopathyDataset'
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    patience = 7
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Early Stopping Patience: {patience}")
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Train the model
    model, history = train_model(
        base_path=base_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        save_dir='model'
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

