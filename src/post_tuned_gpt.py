import torch
import json
import yaml
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lora_transformer import load_lora_model

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")



def run_post_trained_gpt2(cfg:dict):
    
    
    print(cfg)
    rank = cfg.get("rank",8)
    alpha = cfg.get("alpha", None)
    checkpoint_path = cfg.get("checkpoint_path", None)
    experiment_name = cfg["experiment_name"]
    # Load our saved dataset sample
    with open("./experiments/data_sample.json", "r") as f:
        samples = json.load(f)

    with open(f"./experiments/{experiment_name}_gpt2.txt", "w") as f:
        f.write(f"=== {experiment_name.upper()} GPT-2 OUTPUTS (FINE-TUNED) ===\n\n")
        
        for i, item in enumerate(samples):
            prompt = (
                f"### Instruction:\n"
                f"{item['instruction']}\n\n"
                f"### Response:\n"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            model = load_lora_model("gpt2",rank,alpha,checkpoint_path,"cpu")
            outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)# type:ignore
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            f.write(f"--- SAMPLE {i+1} ---\n")
            f.write(f"EXPECTED TARGET:\n{item['output']}\n\n")
            f.write(f"GPT-2 {experiment_name.upper()} OUTPUT:\n{generated_text}\n")
            f.write("\n" + "="*40 + "\n\n")

    print(f"Baseline outputs recorded in experiments/{experiment_name}_gpt2.txt")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_post_trained_gpt2(cfg)
