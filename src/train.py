import torch
import torch.optim as optim
import time
import json
import yaml
import argparse
import os
from transformers import AutoModelForCausalLM

from src.lora_transformer import inject_lora,count_trainable_params
from src.data import get_dataloaders
from transformers import PreTrainedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peak_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    mem = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    return mem

gpt2_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("gpt2")




def run_epoch(model, dataloader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total, steps = 0.0, 0, 0, 0
    start_time = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            _, indices = torch.max(shift_logits, 2)
            ignore_mask = shift_labels != -100

            correct += ((indices == shift_labels) & ignore_mask).sum().item()
            total += ignore_mask.sum().item()
            steps += 1

    elapsed = time.time() - start_time
    return {
        "loss": running_loss / steps,
        "accuracy": 100 * correct / total,
        "time_per_epoch": elapsed,
        "avg_time_per_step": elapsed / steps,
    }
    

def fine_tune_gpt2(cfg: dict):
    
    
    use_lora = cfg.get("use_lora", True)
    rank = cfg.get("rank", 8)
    alpha = cfg.get("alpha", None)
    lr = cfg.get("lr", 0.001 if use_lora else 3e-5)
    epochs = cfg.get("epochs", 3)
    batch_size = cfg.get("batch_size", 8)
    early_stop_patience = cfg.get("early_stop_patience", 5)
    data_path = cfg["data_path"]
    num_examples = cfg.get("num_examples", 10000)
    model_name = cfg.get("model_name", "gpt2")
    experiment_name = cfg["experiment_name"]
    output_dir = cfg.get("output_dir", "experiments")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{experiment_name}_best_model.pt")
    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    if use_lora:
        model = inject_lora(model, rank=rank, alpha=alpha)
    else:
        for p in model.parameters():
            p.requires_grad = True
    model.to(device)#type:ignore
    
    trainable_params, total_params = count_trainable_params(model)
    print(f"[{experiment_name}] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")
   
    
    # Setup Optimizer (only train what requires gradients)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)
    
    # Get Data
    
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(
        data_path=data_path, batch_size=batch_size, num_examples=num_examples
    )
    print(f"[{experiment_name}] Training batches: {len(train_dataloader)} | "
          f"Validation batches: {len(val_dataloader)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"Starting training on {device}...")
    print(f"Training batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")

    # ==========================================
    # YOU WRITE THE TRAINING LOOP DOWN HERE
    # ==========================================
    
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_dataloader, optimizer=optimizer)
        val_metrics = run_epoch(model, val_dataloader, optimizer=None)
        mem_mb = peak_gpu_memory_mb()
        start_time = time.time()
        record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_time_sec": train_metrics["time_per_epoch"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_time_sec": val_metrics["time_per_epoch"],
                "peak_gpu_memory_mb": mem_mb,
                "trainable_params": trainable_params,
                "total_params": total_params,
            }
        history.append(record)

        print(f"[{experiment_name}] Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {record['train_loss']:.4f} | Train Acc: {record['train_accuracy']:.2f}% | "
            f"Val Loss: {record['val_loss']:.4f} | Val Acc: {record['val_accuracy']:.2f}% | "
            f"GPU Mem: {mem_mb}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            lora_state_dict = {
                name: param
                for name, param in model.state_dict().items()
                if model.get_parameter(name).requires_grad
            }
            torch.save(lora_state_dict, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[{experiment_name}] Early stopping at epoch {epoch + 1}")
                break

        summary = {
            "experiment_name": experiment_name,
            "use_lora": use_lora,
            "rank": rank if use_lora else None,
            "alpha": alpha if use_lora else None,
            "lr": lr,
            "epochs_run": len(history),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "final_val_accuracy": history[-1]["val_accuracy"],
            "final_val_loss": history[-1]["val_loss"],
            "avg_train_time_sec": sum(h["train_time_sec"] for h in history) / len(history),
            "peak_gpu_memory_mb": max((h["peak_gpu_memory_mb"] or 0) for h in history) or None,
            "history": history,
        }

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[{experiment_name}] Saved results to {results_path}")

        return model, summary

                                
                
    # return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fine_tune_gpt2(cfg)
