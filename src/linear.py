
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self,in_features,out_features ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(in_features,out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self,x):
        output = torch.matmul(x, self.weight) + self.bias
        
        
        return output