import torch
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq,DataCollatorForLanguageModeling

# Initialize tokenizer once for the module
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def format_prompt(data_row):
    

    instruction = data_row["instruction"]
    input_text = data_row.get("input", "")
    output = data_row["output"]
    
    print(f"this is the data row {data_row}")
    
    print(f"this is input text {input_text}")
    
    print(f"this is instruction{instruction}")
    
    print(f"this is output text {output}")

    if input_text and input_text.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def tokenize_function(data_row):
    text_string = format_prompt(data_row)
    token_output = tokenizer(text_string, truncation=True, max_length=256)
    #token_output["labels"] = token_output["input_ids"]
    return token_output

def get_dataloaders(data_path="./experiments/training_data.jsonl", batch_size=4,num_examples=1000):
    """Loads local data, tokenizes, splits, and returns train/val DataLoaders."""
    
    # 1. Load your LOCAL data
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    
    n = min(num_examples,len(raw_dataset))
    
    raw_dataset = raw_dataset.select(range(n))

    # 2. Tokenize the dataset
    tokenized_rows = []
    for row in raw_dataset:
        tokenized_row = tokenize_function(row)
        
        print(f"This is the tokenized row {tokenized_row}")
        tokenized_rows.append(tokenized_row)
        
    tokenized_dataset = Dataset.from_list(tokenized_rows)

    # 3. Split into Train (90%) and Validation (10%)
    split_data = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    # 4. Set up the Collator 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        split_data["train"],#type:ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_dataloader = DataLoader(
        split_data["test"],#type:ignore
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return train_dataloader, val_dataloader,tokenizer


if __name__ == "__main__":
    print("Loading dataloaders...")
    
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(
        data_path="./experiments/training_data.jsonl", batch_size=1, num_examples=10
    )
   
    
  
    
    print("checkup is done...")
    