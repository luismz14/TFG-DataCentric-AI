import torch
import torch.nn as nn
from torchvision import models

class PolypClassifier(nn.Module):
    def __init__(self, num_classes: int):
        """
        Inicializa el modelo baseline usando EfficientNet-B0 pre-entrenado.
        """
        super(PolypClassifier, self).__init__()
        
        # 1. Load the pre-trained EfficientNet-B0 model
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 2. Fix the backbone layers to prevent them from updating during training
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Modify the classifier to fit our number of classes
        in_features = self.backbone.classifier[1].in_features
        # Replace the last layer of the classifier with a new one that has the correct number of output classes
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 4. Unfreeze the classifier layers to allow them to be trained
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define cómo fluyen los datos (tensores) a través del modelo.
        """
        return self.backbone(x)