import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM,PreTrainedModel
from src.lora_gpt_layer import LoRAGPTLayer


def inject_lora(model)->PreTrainedModel:
    modules_list = list(model.named_modules())
    for name, module in modules_list:
        if(name.endswith("c_attn")):
            #print(f"This is the name: {name}, this is the module: {module}")
            
            child = name.split(".")[-1]
            parent = ".".join(name.split(".")[:-1])
            
            loralayer = LoRAGPTLayer(module.nx,module.nf,module.weight,module.bias,8)
            
            parent_module = model.get_submodule(parent)
            
            setattr(parent_module,child,loralayer)
            
         
    # trainable_params = 0
    # frozen_params = 0
            
    # for p in model.parameters():
    #     if p.requires_grad:
    #         trainable_params += p.numel()
    #     else:
    #         frozen_params += p.numel()
            
    # print(f"Trainable Parameters (LoRA): {trainable_params:,}")
    # print(f"Frozen Parameters: {frozen_params:,}")
    # print(f"Total Parameters: {trainable_params + frozen_params:,}")
    
    return model
            
           
        
        
        




        
if __name__ == "__main__":
    print("Loading GPT-2...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    
    for param in gpt2_model.parameters():
        param.requires_grad = False
        
   
    
  
    
    print("Inspecting layers...")
    inject_lora(gpt2_model)