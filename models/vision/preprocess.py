"""Vision model preprocessing utilities."""
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Union, Tuple, Optional


class VisionPreprocessor:
    def __init__(self, 
                 input_size: Union[int, Tuple[int, int]] = 224,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 model_type: str = "resnet"):
        
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        
        # Default ImageNet normalization
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
            
        self.input_size = input_size
        self.model_type = model_type
        
        # Create preprocessing pipeline
        if model_type == "vit":
            # Vision Transformer preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # Standard CNN preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """Preprocess single image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
            
        return self.transform(image).unsqueeze(0)
    
    def preprocess_batch(self, images: List[Union[Image.Image, str]]) -> torch.Tensor:
        """Preprocess batch of images.
        
        Args:
            images: List of PIL Images or image paths
            
        Returns:
            Preprocessed images tensor [batch_size, 3, H, W]
        """
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert('RGB')
                
            processed_images.append(self.transform(img))
        
        return torch.stack(processed_images)
    
    def get_config(self) -> dict:
        """Get preprocessing configuration."""
        return {
            "input_size": self.input_size,
            "model_type": self.model_type,
            "transform": str(self.transform)
        }