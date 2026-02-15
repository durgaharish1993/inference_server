"""Vision model for image classification/feature extraction."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import torchvision.models as models


class VisionModel(nn.Module):
    def __init__(self, 
                 model_name: str = "resnet50",
                 num_classes: Optional[int] = None,
                 pretrained: bool = True,
                 feature_extraction: bool = True):
        super().__init__()
        
        # Load backbone model
        if model_name.startswith("resnet"):
            self.backbone = getattr(models, model_name)(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            
            if feature_extraction:
                # Remove classification head for feature extraction
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                self.feature_dim = feature_dim
            else:
                # Keep or replace classification head
                if num_classes and num_classes != 1000:
                    self.backbone.fc = nn.Linear(feature_dim, num_classes)
                self.feature_dim = num_classes or 1000
                
        elif model_name.startswith("vit"):
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(f"google/{model_name}")
            self.feature_dim = self.backbone.config.hidden_size
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.model_name = model_name
        self.feature_extraction = feature_extraction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision model.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            
        Returns:
            Features or logits [batch_size, feature_dim]
        """
        if self.model_name.startswith("vit"):
            outputs = self.backbone(pixel_values=x)
            return outputs.pooler_output
        else:
            features = self.backbone(x)
            if self.feature_extraction:
                return features.flatten(1)  # [batch_size, feature_dim]
            return features
    
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        self.eval()
        return self.forward(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for ONNX export."""
        return {
            "model_name": f"vision_{self.model_name}",
            "input_shape": [3, 224, 224],
            "output_shape": [self.feature_dim],
            "dynamic_axes": {
                "images": {0: "batch_size"},
                "features": {0: "batch_size"}
            },
            "feature_extraction": self.feature_extraction
        }