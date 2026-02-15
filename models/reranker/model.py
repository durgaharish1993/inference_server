"""Reranker model for document ranking."""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer


class RerankerModel(nn.Module):
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        
        # Load pre-trained transformer model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add classification head for ranking
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Single score output
        )
        
        self.device = device
        self.to(device)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for reranker.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            
        Returns:
            Relevance scores [batch_size, 1]
        """
        # Get transformer outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Generate relevance score
        scores = self.classifier(cls_output)  # [batch_size, 1]
        
        return scores
    
    def encode_pairs(self, 
                    queries: List[str], 
                    documents: List[str],
                    max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode query-document pairs for ranking.
        
        Args:
            queries: List of query strings
            documents: List of document strings  
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Create query-document pairs
        pairs = []
        for query, doc in zip(queries, documents):
            pairs.append(f"{query} [SEP] {doc}")
        
        # Tokenize pairs
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).to(self.device)
        }
    
    @torch.no_grad()
    def rank_documents(self, 
                      query: str, 
                      documents: List[str],
                      return_scores: bool = True) -> List[Tuple[int, float]]:
        """Rank documents for a given query.
        
        Args:
            query: Query string
            documents: List of document strings
            return_scores: Whether to return scores with indices
            
        Returns:
            List of (doc_index, score) tuples sorted by relevance
        """
        self.eval()
        
        # Encode query-document pairs
        queries = [query] * len(documents)
        inputs = self.encode_pairs(queries, documents)
        
        # Get relevance scores
        scores = self.forward(**inputs)
        scores = scores.squeeze(-1).cpu().numpy()  # [batch_size]
        
        # Sort by scores (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        if return_scores:
            return [(idx, float(scores[idx])) for idx in ranked_indices]
        else:
            return ranked_indices
    
    def get_config(self) -> Dict[str, any]:
        """Get model configuration for ONNX export."""
        return {
            "model_name": "reranker",
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "output_names": ["relevance_scores"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "token_type_ids": {0: "batch_size", 1: "sequence"},
                "relevance_scores": {0: "batch_size"}
            }
        }