"""CLIP image encoder model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import torchvision.transforms as transforms
from PIL import Image


class CLIPImageEncoder(nn.Module):
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        from transformers import CLIPVisionModel
        
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP image encoder.
        
        Args:
            pixel_values: Preprocessed images tensor [batch_size, 3, 224, 224]
            
        Returns:
            Image embeddings [batch_size, hidden_size]
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        self.eval()
        return self.forward(images)
    
    def get_config(self) -> dict:
        """Get model configuration for ONNX export."""
        return {
            "model_name": "clip_image_encoder",
            "input_shape": [3, 224, 224],
            "output_shape": [512],  # CLIP base hidden size
            "dynamic_axes": {
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"}
            }
        }