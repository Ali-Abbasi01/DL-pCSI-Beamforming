#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple
import json

#############################
# HYPERPARAMETERS / CONFIG #
#############################
CSV_PATH = "../data/synthesis/synthesized_data_UIU.csv"
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
HIDDEN_DIM = 128
SEED = 42
SAVE_FIG_PATH = "loss_plot.png"  # Where to save the train/val plot

#####################
# DATA LOADING LOGIC
#####################
def parse_matrix(str_rep: str) -> np.ndarray:
    """
    Example parser that assumes the matrix is stored in JSON with 'real' as a 2D list
    or stored as a Python literal like [[1.0, 2.0], [3.0, 4.0]].
    Adjust if your CSV uses a different format!
    """
    # Minimal approach: try JSON, if it fails, try eval
    try:
        data = json.loads(str_rep)
        # If your CSV is purely real, you might just do: return np.array(data)
        # but here we handle a possible key 'real' if it exists:
        if isinstance(data, dict) and "real" in data:
            return np.array(data["real"], dtype=np.float32)
        else:
            return np.array(data, dtype=np.float32)
    except:
        # Fallback if not valid JSON: use eval to parse Python literal
        # WARNING: using eval can be unsafe if strings are not trusted.
        return np.array(eval(str_rep), dtype=np.float32)

def load_dataset(csv_path: str):
    """
    Loads the dataset from CSV, returning:
      H_list: list of channel matrices (real)
      Sigma_list: list of covariance (target) matrices (real)
    We'll compute Sigma = V @ P @ V.T from the second and third columns.
    """
    df = pd.read_csv(csv_path)

    # Adapt these column names to match your CSV structure
    # Here we assume columns named "channel", "bf_matrix", and "p_allocation"
    df["U_R"] = df["U_R"].apply(lambda x: torch.tensor(json.loads(x)))

    df["U_T"] = df["U_T"].apply(lambda x: torch.tensor(json.loads(x)))

    df["H_bar"] = df["H_bar"].apply(lambda x: torch.tensor(json.loads(x)))

    df ["G"] = df["G"].apply(lambda x: torch.tensor(json.loads(x)))

    df["V"] = df["V"].apply(lambda x: torch.tensor(json.loads(x)))

    df["P"] = df["P"].apply(lambda x: torch.tensor(json.loads(x)))

    H_list = []
    Sigma_list = []

    # Compute Sigma from V, P for each row
    for idx, row in df.iterrows():
        H = row["channel"]      # shape: (m, n)
        V = row["V"]    # shape: (n, n) or similar
        P = row["P"] # shape: (n, n) or vector (n,) etc.
        # If P is diagonal, you might need e.g. np.diag(P) here:
        # P = np.diag(P) if it's stored as just the diagonal
        Sigma = V @ P @ V.T
        H_list.append(H)
        Sigma_list.append(Sigma)

    return H_list, Sigma_list

def train_val_split(H_list, Sigma_list, val_ratio=VALIDATION_SPLIT, seed=SEED):
    """
    Splits the data into train and validation sets.
    """
    N = len(H_list)
    indices = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_idx = int(N*(1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    H_train = [H_list[i] for i in train_idx]
    Sigma_train = [Sigma_list[i] for i in train_idx]
    H_val = [H_list[i] for i in val_idx]
    Sigma_val = [Sigma_list[i] for i in val_idx]

    return H_train, Sigma_train, H_val, Sigma_val

def flatten_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Flatten a real-valued 2D matrix to 1D array.
    """
    return mat.flatten()

class SigmaDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for H -> Sigma mapping.
    Each item is (H_in, Sigma_label), where both are flattened real vectors.
    """
    def __init__(self, H_list, Sigma_list):
        self.X = []
        self.y = []
        for H, Sigma in zip(H_list, Sigma_list):
            x = flatten_matrix(H)
            t = flatten_matrix(Sigma)
            self.X.append(x)
            self.y.append(t)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

##################
# MODEL DEFINITION
##################
class SigmaPredictor(nn.Module):
    """
    A simple feedforward network that maps a flattened real channel vector
    to a flattened real Sigma vector.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=HIDDEN_DIM):
        super(SigmaPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out

#########################
# TRAINING & EVAL LOGIC
#########################
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(model.device)
                y_batch = y_batch.to(model.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path=SAVE_FIG_PATH):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")

def evaluate_dataset(model, loader):
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

def main():
    # 1) Load the dataset
    print(f"Loading dataset from {CSV_PATH} ...")
    H_list, Sigma_list = load_dataset(CSV_PATH)

    # 2) Split train/val
    H_train, Sigma_train, H_val, Sigma_val = train_val_split(
        H_list, Sigma_list,
        val_ratio=VALIDATION_SPLIT, seed=SEED
    )

    # 3) Build Dataset objects and DataLoaders
    train_ds = SigmaDataset(H_train, Sigma_train)
    val_ds = SigmaDataset(H_val, Sigma_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Build the model
    #   Let H be shape (m, n). Flatten -> m*n
    #   Then we feed in a 1D real vector of size m*n as input.
    #   Sigma is shape (n, n), flattened -> n*n
    sample_H = H_train[0]
    m, n = sample_H.shape
    in_dim = m * n
    out_dim = n * n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SigmaPredictor(in_dim, out_dim, hidden_dim=HIDDEN_DIM)
    model.device = device
    model.to(device)

    # 5) Train
    print(f"Starting training for {EPOCHS} epochs ...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE
    )

    # 6) Plot training vs. validation performance
    plot_losses(train_losses, val_losses, SAVE_FIG_PATH)

    # 7) Final evaluation
    train_mse = evaluate_dataset(model, train_loader)
    val_mse = evaluate_dataset(model, val_loader)
    print(f"Final Train MSE: {train_mse:.6f}")
    print(f"Final Validation MSE: {val_mse:.6f}")

    # 8) Save the model
    torch.save(model.state_dict(), "final_sigma_model.pt")
    print("Model weights saved to final_sigma_model.pt")


if __name__ == "__main__":
    main()