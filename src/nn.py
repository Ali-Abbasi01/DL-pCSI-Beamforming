# notebooks/modeling/train_sigma_predictor.ipynb

import sys
sys.path.append('../../src')  # so we can import from src/ easily

import numpy as np
import torch
import torch.optim as optim
from data_processing import load_channel_sigma_data, train_test_split
from neural_networks import SigmaPredictor

# 1) Load data
file_path = '../../data/processed/channel_sigma.npz'
H, Sigma_opt = load_channel_sigma_data(file_path)  # shapes (N, m, n), (N, n, n)

# Flatten the channel for a simple approach
N, m, n = H.shape
H_flat = H.reshape(N, -1).real  # ignoring imaginary part for demonstration

# Convert to torch tensors
H_tensor = torch.from_numpy(H_flat).float()
Sigma_tensor = torch.from_numpy(Sigma_opt.real).float()  # ignoring imaginary

# 2) Split train/test
split_ratio = 0.2
idx_split = int(N*(1.0 - split_ratio))
H_train, H_test = H_tensor[:idx_split], H_tensor[idx_split:]
Sigma_train, Sigma_test = Sigma_tensor[:idx_split], Sigma_tensor[idx_split:]

# 3) Create DataLoaders
train_ds = torch.utils.data.TensorDataset(H_train, Sigma_train)
test_ds = torch.utils.data.TensorDataset(H_test, Sigma_test)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 4) Define model, optimizer
model = SigmaPredictor(m, n, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5) Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_H, batch_Sigma in train_loader:
        # Forward pass
        Sigma_pred = model(batch_H)
        # MSE between predicted Sigma and ground truth
        loss = torch.mean((Sigma_pred - batch_Sigma)**2)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_H)
    
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 6) Simple test set evaluation
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_H, batch_Sigma in test_loader:
        Sigma_pred = model(batch_H)
        loss = torch.mean((Sigma_pred - batch_Sigma)**2)
        test_loss += loss.item() * len(batch_H)

    test_loss /= len(test_loader.dataset)
    print("Test MSE:", test_loss)
