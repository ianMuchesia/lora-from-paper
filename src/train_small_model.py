import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from src.small_model import ToyModel










# 2. Create some dummy data (Batch of 16)
x = torch.randn(16, 20)


# Dummy targets (binary classification: 0 or 1)
targets = torch.randint(0, 2, (16,))

print(targets)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


criterion =  nn.CrossEntropyLoss()




epochs = 10

ranks = [4,8,16]


for rank in ranks:
    

    # 1. Instantiate the model
    model = ToyModel(in_features=20, out_features=2, hidden_features=64, rank=rank)

    # Filter for only parameters where p.requires_grad is True
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Pass the filtered iterator to the Adam optimizer
    optimizer = optim.Adam(trainable_params, lr=0.001)

    accuracy = 0
    training_loss = 0

    for epoch in range(epochs):
        
        model.train()
        
        optimizer.zero_grad()
        
        output = model(x)
        
        print(f"This is the output shape: {output.shape}")
        print(f"This is the target shape: {targets.shape}")
        
        
        loss = criterion(output,targets)
        
        loss.backward()
        
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | ")
        
    
        preds = torch.argmax(output, dim=1)
        
        # Sum the correct predictions
        correct = (preds == targets).sum().item()

        # Explicitly divide by the total number of elements (16)
        total = targets.size(0)
        accuracy = correct / total
        
        training_loss = loss.item()
        
    
    trainable_params = 0
    frozen_params = 0
    
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            frozen_params += p.numel()
            
    print(f"Trainable Parameters (LoRA): {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")
    print(f"Total Parameters: {trainable_params + frozen_params:,}")
    
    with open(f"./experiments/rank_{rank}_results.txt", "w") as f:
        # 1. Write the table header with column widths
        f.write(f"Frozen Params {frozen_params:,}\n")
        f.write(f"Trainable Params {trainable_params:,}\n")
        f.write(f"Total Params {trainable_params + frozen_params:,}\n")
        f.write("\n\n")
        f.write(f"Final Training Loss {training_loss:.4f}\n")
        f.write(f"Final Accuracy Performance {100 * accuracy:.2f}%\n")

    
    


