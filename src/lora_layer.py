
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self,in_features,out_features,rank):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn(in_features,out_features))
        self.W.requires_grad = False
        
        self.A = nn.Parameter(torch.randn(rank,out_features))
        
        self.B = nn.Parameter(torch.zeros(in_features,rank))
        
    
    
    def forward(self,x):
        # y = xw + xba
        output =  torch.matmul(x,self.W) + (x@self.B) @ self.A  
        
        
        return output