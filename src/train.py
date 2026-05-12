import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from dataset import EssayDataset
from model import ECE364Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Reproducability Seed, chose 42 cus 6 7 (might be brainrotted)
torch.manual_seed(42)
np.random.seed(42)

# Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Setuo Tokenizer and Model
model_name = "nreimers/MiniLM-L6-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ECE364Classifier(model_name).to(device)

# Check 15M parameter compliance
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")

# Load Datasets
train_path = 'llm-or-not/dataset/train.csv'
test_path = 'llm-or-not/dataset/test.csv'

# Create internal Validation Split (10%) to check accuracy
full_train_dataset = EssayDataset(train_path, tokenizer, max_length=256)
train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.1, random_state=42)

train_loader = DataLoader(Subset(full_train_dataset, train_indices), batch_size=16, shuffle=True)
val_loader = DataLoader(Subset(full_train_dataset, val_indices), batch_size=16, shuffle=False)

test_dataset = EssayDataset(test_path, tokenizer, max_length=256, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = nn.BCELoss()

# Lists to store metrics for plotting
epoch_list = []
loss_history = []
auc_history = []

# Training Loop
epochs = 3 
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        probs = model(ids, mask).squeeze()
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ROC-AUC Validation
    model.eval()
    val_labels, val_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            probs = model(ids, mask).squeeze().cpu().numpy()
            val_labels.extend(labels)
            val_preds.extend(probs)
    
    auc = roc_auc_score(val_labels, val_preds)
    avg_loss = epoch_loss / len(train_loader)
    
    # Store metrics
    epoch_list.append(epoch + 1)
    loss_history.append(avg_loss)
    auc_history.append(auc)
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val ROC-AUC: {auc:.4f}")

# Plotting Logic
print("Generating training metrics graph...")
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss (BCE)', color=color)
ax1.plot(epoch_list, loss_history, marker='o', color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation ROC-AUC', color=color)
ax2.plot(epoch_list, auc_history, marker='s', color=color, label='ROC-AUC')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss vs. Validation ROC-AUC')
plt.grid(True, linestyle='--')
plt.savefig('training_metrics.png')
print("Graph saved as training_metrics.png")

# Final Submission Generation
print("Generating prediction.csv...")
model.eval()
results = []
with torch.no_grad():
    for batch in test_loader:
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        probs = model(ids, mask).squeeze().cpu().numpy()
        results.extend(zip(batch['id'], probs))

submission = pd.DataFrame(results, columns=['Id', 'Prob'])
submission.to_csv('prediction.csv', index=False)
print("Done! prediction.csv has been generated.")