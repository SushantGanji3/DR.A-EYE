"""
PyTorch Dataset and DataLoader for Diabetic Retinopathy images
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DiabeticRetinopathyDataset(Dataset):
    """Custom Dataset for Diabetic Retinopathy images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of integer labels (0-4)
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def get_transforms(train=True):
    """Get data transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def prepare_data_splits(base_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare train/validation/test splits from the dataset
    
    Args:
        base_path: Path to DiabeticRetinopathyDataset folder
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    base_path = Path(base_path)
    images_dir = base_path / 'gaussian_filtered_images' / 'gaussian_filtered_images'
    train_csv = base_path / 'train.csv'
    
    # Class mapping
    class_mapping = {
        0: 'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR'
    }
    
    # Load CSV
    df = pd.read_csv(train_csv)
    
    # Build image paths and labels
    # Since images are organized by class folders, collect all images from each class
    image_paths = []
    labels = []
    
    # Create a mapping from id_code to diagnosis for reference
    id_to_diagnosis = dict(zip(df['id_code'], df['diagnosis']))
    
    # Collect all images from each class folder
    for diagnosis, class_name in class_mapping.items():
        class_dir = images_dir / class_name
        if class_dir.exists():
            # Get all PNG images in this class folder
            class_images = list(class_dir.glob('*.png'))
            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(diagnosis)
    
    print(f"Total images collected: {len(image_paths)}")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_dataloaders(base_path, batch_size=32, num_workers=4, 
                   test_size=0.15, val_size=0.15, random_state=42):
    """
    Get DataLoaders for train, validation, and test sets
    
    Args:
        base_path: Path to DiabeticRetinopathyDataset folder
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Prepare splits
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_data_splits(base_path, test_size, val_size, random_state)
    
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        train_paths, train_labels, transform=get_transforms(train=True)
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_paths, val_labels, transform=get_transforms(train=False)
    )
    test_dataset = DiabeticRetinopathyDataset(
        test_paths, test_labels, transform=get_transforms(train=False)
    )
    
    # Create dataloaders
    # pin_memory only helps with GPU training, disable for CPU
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader

