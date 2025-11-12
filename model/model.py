"""
ResNet-18 model for Diabetic Retinopathy classification
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class DiabeticRetinopathyModel(nn.Module):
    """Fine-tuned ResNet-18 for Diabetic Retinopathy classification"""
    
    def __init__(self, num_classes=5, pretrained=True):
        """
        Args:
            num_classes: Number of output classes (default: 5)
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(DiabeticRetinopathyModel, self).__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True

