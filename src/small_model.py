
import torch
import torch.nn as nn

from src.linear import Linear
from src.lora_layer import LoRALayer


class ToyModel(nn.Module):
    def __init__(self,in_features,out_features,hidden_features,rank):
        super().__init__()
        
        self.mlp = Linear(in_features,hidden_features)
        self.LoRA = LoRALayer(hidden_features,out_features,rank)
        self.relu = nn.ReLU()
        
        
        
    def forward(self,x):
        
        out = self.mlp(x)
        
        out = self.relu(out)
        
        out = self.LoRA(out)
        
        return out  