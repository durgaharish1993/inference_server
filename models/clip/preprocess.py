"""CLIP preprocessing utilities."""
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Union, Dict, Any
from transformers import CLIPProcessor, CLIPTokenizer


class CLIPPreprocessor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Standard CLIP image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def preprocess_images(self, images: Union[List[Image.Image], List[str]]) -> torch.Tensor:
        """Preprocess images for CLIP image encoder.
        
        Args:
            images: List of PIL Images or image paths
            
        Returns:
            Preprocessed images tensor [batch_size, 3, 224, 224]
        """
        if isinstance(images[0], str):
            images = [Image.open(img_path).convert('RGB') for img_path in images]
        
        processed_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert('RGB')
            processed_images.append(self.image_transform(img))
        
        return torch.stack(processed_images)
    
    def preprocess_texts(self, texts: List[str], max_length: int = 77) -> Dict[str, torch.Tensor]:
        """Preprocess texts for CLIP text encoder.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def process_batch(self, images: List[Image.Image], texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process both images and texts in a batch."""
        processed = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return processed