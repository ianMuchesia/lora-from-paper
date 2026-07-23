import json
from datasets import load_dataset

# Load raw dataset
ds = load_dataset("yahma/alpaca-cleaned", split="train")

# Grab first 5 examples
sample_data = [ds[i] for i in range(5)]

# Save to experiments/data_sample.json for inspection
with open("./experiments/data_sample.json", "w") as f:
    json.dump(sample_data, f, indent=4)
    
# FIX: Convert the Dataset object into a list of dictionaries before saving
with open("./experiments/training_data.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row) + "\n")


print("Saved 5 raw dataset samples to experiments/data_sample.json")
print("Saved entire dataset to experiments/training_data.json")
