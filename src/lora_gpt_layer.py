
import torch
import torch.nn as nn


class LoRAGPTLayer(nn.Module):
    def __init__(self,in_features,out_features,original_weight,orignial_bias,rank,alpha=None):
        super().__init__()
        
        self.W = original_weight
        self.W.requires_grad = False
        
        # self.bias = nn.Parameter(torch.zeros(out_features))
        self.bias = orignial_bias
        if self.bias is not None:
            self.bias.requires_grad = False
            
            
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        
        self.A = nn.Parameter(torch.randn(rank,out_features))
        
        self.B = nn.Parameter(torch.zeros(in_features,rank))
        self.merged = False
        
    
    
    def forward(self,x):
        # y = xw + xba
        if self.merged:
            output = torch.matmul(x,self.W)
        else:
            output =  torch.matmul(x,self.W) + self.scaling * (x@self.B) @ self.A  
            
        if self.bias is not None:
            output = output + self.bias
        
        
        return output
    
    
    def merge_weights(self):
        if self.merged: return self.W
        
        with torch.no_grad():
            self.W += self.scaling * (self.B @ self.A)
            self.merged = True
            
            return self.W
        
    def unmerge_weights(self):
        if not self.merged: return self.W
        with torch.no_grad():
            self.W -= self.scaling * (self.B @ self.A)
            
            self.merged = False
            
            return self.W
        
        
        
