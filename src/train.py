import torch
import torch.optim as optim
import time
import json
from transformers import AutoModelForCausalLM

from src.lora_transformer import inject_lora
from src.data import get_dataloaders
from transformers import PreTrainedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("gpt2")

def fine_tune_gpt2(use_lora: bool):

    model: PreTrainedModel = inject_lora(model=gpt2_model) if use_lora else gpt2_model
    
    model.to(device) #type:ignore
   
    
    # Setup Optimizer (only train what requires gradients)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)
    
    # Get Data
    train_dataloader, val_dataloader = get_dataloaders(
        data_path="./experiments/training_data.jsonl", 
        batch_size=1
    )
    
    print(f"Starting training on {device}...")
    print(f"Training batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")

    # ==========================================
    # YOU WRITE THE TRAINING LOOP DOWN HERE
    # ==========================================
    
    epochs = 3
    best_val_loss  = 0
    early_stop_patience = 5
    
    history = []
    
    patience_counter = 0
    
    for epoch in range(epochs):
        training_loss = 0
        correct = 0
        total_steps = 0
        total = 0
        
        start_time = time.time()
        model.train()
                    
        for batch in train_dataloader:
            
           
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            print(f"this is x shape {input_ids.shape}")
            print(f"this is y shape {labels.shape}")
            
            
            outputs = model(
                       input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels
                    )
            
            
            print(f"this is the shape of the output {outputs.shape}")
            
            
            loss = outputs.loss
            
            loss.backward()
            
            
            optimizer.step()
            
            training_loss += loss.item()
            
            _,indices = torch.max(outputs.data,2)
            
            correct_guesses = (indices == labels)
            
            correct += correct_guesses.sum().item()
            
            
            
            
            total_steps += 1
            total += labels.numel()
            print(f"this is the loss {loss.item()}")
            
            
        end_time = time.time()
        total_time = end_time - start_time
        average_training_time = total_time/total_steps
        
        
    
        average_training_loss = training_loss/total_steps
        
        training_accuracy = 100 * (correct/total)
        
        print(f"this is the total correct: {correct}")
        print(f"this is the total steps: {total_steps}")
        print(f"this is the total expected labels: {total}")
        print(f"this is the total len of training dataloader: {len(train_dataloader)}")
        print(f"this is the total running loss: {training_loss}")
        print(f"this is the average training loss: {average_training_loss}")
        print(f"this is the average training accuracy: {training_accuracy}")
        
        
        model.eval()
        
        total_time = 0
        val_loss = 0
        
        running_loss = 0
        correct = 0
        
        total_steps = 0
        total =0
        
        
        with torch.no_grad():
            
            for batch in val_dataloader:
                
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                
                
                outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask,
                            labels=labels
                        )
                
                
                loss = outputs.loss
                            
                
                
                
                
                
                training_loss += loss.item()
                
                _,indices = torch.max(outputs.data,2)
                
                correct_guesses = (indices == labels)
                
                correct += correct_guesses.sum().item()
                
                
                
                
                total_steps += 1
                total += labels.numel()
                print(f"this is the loss {loss.item()}")
                
            end_time = time.time()
        
        
            total_time = end_time - start_time
            
         
        average_validation_loss = running_loss/total_steps
        average_validation_time = total_time/total_steps
    
        validation_accuracy = 100 * (correct/total)
        
        
        print(f"this is the total correct: {correct}")
        print(f"this is the total steps: {total_steps}")
        print(f"this is the total expected labels: {total}")
        print(f"this is the total len of validation dataloader: {len(val_dataloader)}")
        print(f"this is the total running loss: {training_loss}")
        print(f"this is the average validation loss: {average_validation_loss}")
        print(f"this is the average validation accuracy: {validation_accuracy}")
        print(f"\n")
        
        metrics = {
            
                "Epoch":epoch + 1,
                "train_loss": average_training_loss,
                "training_acc": training_accuracy,
                "training_time":average_training_time,
                "val_loss":average_validation_loss,
                "val_time": average_validation_time,
                "val_accuracy": validation_accuracy,
                
            }
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {average_training_loss:.4f} | Train Accuracy: {training_accuracy:.2f}%")
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {average_validation_loss:.4f} | Val Accuracy: {validation_accuracy:.2f}%")

            
        history.append(metrics)
        
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            patience_counter = 0
            torch.save(model.state_dict(),"./checkpoints/best_gpt_model.pt")
        
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stoppng triggered at epoch {epoch + 1}")
                break
            
        
    with open(f"./experiments/training_data.json","w") as f:
        json.dump(history,f,indent=4)
        
                                
                
    # return model

if __name__ == "__main__":
    print("Initializing fine-tuning pipeline...")
    # Change to True when you want to use LoRA, False for full fine-tuning
    trained_model = fine_tune_gpt2(use_lora=True)