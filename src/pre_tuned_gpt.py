import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load our saved dataset sample
with open("./experiments/data_sample.json", "r") as f:
    samples = json.load(f)

with open("./experiments/baseline_gpt2.txt", "w") as f:
    f.write("=== BASELINE GPT-2 OUTPUTS (UN-TUNED) ===\n\n")
    
    for i, item in enumerate(samples):
        prompt = (
            f"### Instruction:\n"
            f"{item['instruction']}\n\n"
            f"### Response:\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)# type:ignore
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        f.write(f"--- SAMPLE {i+1} ---\n")
        f.write(f"EXPECTED TARGET:\n{item['output']}\n\n")
        f.write(f"GPT-2 BASELINE OUTPUT:\n{generated_text}\n")
        f.write("\n" + "="*40 + "\n\n")

print("Baseline outputs recorded in experiments/baseline_gpt2.txt")