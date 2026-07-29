import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM,PreTrainedModel
from src.lora_gpt_layer import LoRAGPTLayer


def inject_lora(model,rank,alpha)->PreTrainedModel:
    modules_list = list(model.named_modules())
    for name, module in modules_list:
        if(name.endswith("c_attn")):
            #print(f"This is the name: {name}, this is the module: {module}")
            
            child = name.split(".")[-1]
            parent = ".".join(name.split(".")[:-1])
            
            loralayer = LoRAGPTLayer(module.nx,module.nf,module.weight,module.bias,8,alpha)
            
            parent_module = model.get_submodule(parent)
            
            setattr(parent_module,child,loralayer)
            
    return model
            
            
def load_lora_model(model_name, rank,alpha, checkpoint_path, device):

    # Step 1: rebuild the same LoRA structure used during training
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = inject_lora(base_model, rank=rank,alpha=alpha)   # your existing function, unchanged
    model.to(device)

    # Step 2: load only the saved A/B weights into that structure
    lora_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)

    return model
            
            
def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
    
         
    
        
        
        




        
if __name__ == "__main__":
    print("Loading GPT-2...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    
    for param in gpt2_model.parameters():
        param.requires_grad = False
        
   
    
  
    
    print("Inspecting layers...")
    inject_lora(gpt2_model,rank=4,alpha=None)