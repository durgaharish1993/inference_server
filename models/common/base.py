import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def device(self):
        return next(self.parameters()).device

    def to_device(self, device: str):
        return self.to(device)

    def set_eval(self):
        self.eval()
        return self