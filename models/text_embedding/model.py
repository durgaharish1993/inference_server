"""Text embedding model using sentence transformers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Union


class TextEmbeddingModel(nn.Module):
    def __init__(self, 
                 model_name_or_path: str,
                 normalize: bool = True,
                 max_seq_length: int = 512):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        
        # Load the transformer model
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Get hidden size for potential pooling layers
        self.hidden_size = self.transformer.config.hidden_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: Text embeddings [batch_size, hidden_size]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Mean pooling with attention mask
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # Normalize if requested
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings
    
    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling with attention mask."""
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by actual sequence length
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def encode(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        """Encode texts to embeddings (inference mode)."""
        if device is None:
            device = next(self.parameters()).device
            
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)
            
        return embeddings
    
    @classmethod
    def from_sentence_transformers(cls, model_name: str, **kwargs):
        """Load from sentence-transformers model."""
        return cls(model_name, **kwargs)
    
    def get_config(self) -> Dict:
        """Get model configuration for export."""
        return {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
            "hidden_size": self.hidden_size
        }
