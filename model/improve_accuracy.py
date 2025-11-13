"""
Strategies to improve model accuracy from 82% to 97%
This script implements various techniques for better performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import WeightedRandomSampler
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

def create_weighted_sampler(labels):
    """Create weighted sampler for balanced training"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_advanced_transforms(train=True):
    """Advanced data augmentation for better generalization"""
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def get_advanced_optimizer(model, learning_rate=0.0001):
    """Advanced optimizer with better hyperparameters"""
    # Use AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer

def get_advanced_scheduler(optimizer, num_epochs, train_loader):
    """Advanced learning rate scheduler"""
    # OneCycleLR for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    return scheduler

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training strategies summary:
"""
1. Class Imbalance Handling:
   - Use weighted loss (class_weights)
   - Use WeightedRandomSampler
   - Use Focal Loss instead of CrossEntropy

2. Advanced Data Augmentation:
   - RandomCrop with larger size
   - More aggressive color jitter
   - Gaussian blur
   - Random erasing
   - Mixup augmentation

3. Better Training Strategy:
   - Use AdamW optimizer
   - OneCycleLR scheduler
   - Lower initial learning rate (0.0001)
   - Longer training with patience

4. Model Improvements:
   - Try different architectures (ResNet-50, EfficientNet)
   - Ensemble multiple models
   - Use test-time augmentation

5. Regularization:
   - Increase weight decay
   - Add dropout layers
   - Label smoothing

6. Advanced Techniques:
   - Progressive resizing
   - Pseudo-labeling
   - Knowledge distillation
"""

