"""Tokenizer for text embedding models."""
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any


class TextEmbeddingTokenizer:
    def __init__(self, model_name_or_path: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
    
    def encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts for text embedding model."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, torch.Tensor]]:
        """Batch tokenize texts."""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append(self.encode(batch_texts))
        return batches