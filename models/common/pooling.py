import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Mean pooling with attention mask.
    """

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom


class CLS_Pooling(nn.Module):
    """
    CLS token pooling.
    """

    def forward(self, last_hidden_state: torch.Tensor, attention_mask=None):
        return last_hidden_state[:, 0]