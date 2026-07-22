from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model weights (~500MB)...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Done! Model loaded into memory.")