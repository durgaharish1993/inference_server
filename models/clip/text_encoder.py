"""CLIP text encoder model."""
import torch
import torch.nn as nn
from typing import Dict, Any


class CLIPTextEncoder(nn.Module):
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        from transformers import CLIPTextModel
        
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text embeddings [batch_size, hidden_size]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
    
    @torch.no_grad()
    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts to embeddings."""
        self.eval()
        return self.forward(input_ids, attention_mask)
    
    def get_config(self) -> dict:
        """Get model configuration for ONNX export."""
        return {
            "model_name": "clip_text_encoder",
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["text_embeds"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "text_embeds": {0: "batch_size"}
            }
        }