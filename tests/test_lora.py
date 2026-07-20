import torch
from src.lora_layer import LoRALayer

in_features = 512

out_features = 1024

rank = 8




def verify_output_shapes():
    layer = LoRALayer(in_features,out_features,rank)
    
    x = torch.randn(20,in_features)
    
    
    output  = layer.forward(x)
    
    
    assert output.shape == (20,1024), f"output shape mismatch. Expected (20,1024), got {output.shape}"
    
    
def verify_gradients():
    
    layer = LoRALayer(in_features,out_features,rank)
    
    x = torch.randn(20,in_features)
    
    
    output  = layer.forward(x)
    
    loss = output.sum()
    
    loss.backward()
    
    assert layer.A.shape == (rank,out_features), f"layer A shape mismatch. Expected (8,1024), got {layer.A.shape}"
    assert layer.A.grad is not None, "layer A did not receive gradients!"    
    assert layer.B.shape == (in_features,rank), f"output shape mismatch. Expected (512,8), got {layer.B.shape}"
    
    assert layer.B.grad is not None, "layer B did not receive gradients!"    
    assert layer.B.sum() == 0, f"Wrong output , Expected 0 got {layer.B.sum()}"
    
    
    assert layer.W.grad == None ,f"Wrong value for W grad, Expected None , got {layer.W.grad}"
    
    
    
    
    
    
if __name__ == "__main__":
    verify_gradients()
    print("All shape tests passed successfully!")
    