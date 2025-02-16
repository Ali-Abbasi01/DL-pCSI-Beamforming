import os
import json
import torch
import pandas as pd

# Adjust these imports to match your actual module paths
from src.beamforming import wf_algorithm
from src.utils import calculate_rate

def generate_random_mimo_channels(num_samples, n_rx, n_tx, dtype=torch.complex64):
    """
    Generate 'num_samples' random MIMO channels, each of shape (n_rx, n_tx),
    using a complex normal distribution in PyTorch.
    Returns a tensor of shape (num_samples, n_rx, n_tx) with complex entries.
    """
    # Real and imaginary parts ~ N(0,1)
    real_part = torch.randn(num_samples, n_rx, n_tx)
    imag_part = torch.randn(num_samples, n_rx, n_tx)
    channels = torch.complex(real_part, imag_part).to(dtype)
    return channels

def main():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    num_samples = 1000  # Number of random channels
    n_rx = 4            # Number of receive antennas
    n_tx = 4            # Number of transmit antennas

    # -----------------------------
    # 1. Generate random channels
    # -----------------------------
    channels = generate_random_mimo_channels(num_samples, n_rx, n_tx)

    # Prepare a list of data records (one record per channel)
    data_records = []

    # -----------------------------
    # 2. Compute BF matrix, power allocation, and rate for each channel
    # -----------------------------
    for i in range(num_samples):
        ch = channels[i]  # shape: (n_rx, n_tx), complex

        # Instantiate your water-filling or beamforming object/class
        wf = wf_algorithm(ch)  # Adjust constructor as needed

        # 2.1: Get the beamforming matrix (bf_mat) and power allocation (p_alloc)
        bf_mat = wf.bf_matrix()        # shape: (n_tx, ?)
        p_alloc = wf.p_allocation()    # shape: (n_streams,) or (n_tx,)

        # 2.2: Construct the transmit covariance matrix, if needed
        #      Example: Cov = bf_mat @ diag(p_alloc) @ bf_mat^H
        #      (Hermitian transpose in PyTorch: conj().transpose(-2, -1))
        Cov = bf_mat @ torch.diag(p_alloc) @ bf_mat.conj().transpose(-2, -1)

        # 2.3: Calculate the rate
        rate = calculate_rate(ch, Cov)
        # If `rate` is a torch scalar, convert to Python float
        if isinstance(rate, torch.Tensor):
            rate = rate.item()

        # -----------------------------
        # 3. Store record in a list
        # -----------------------------
        # Because CSV doesn't handle complex/binary data directly,
        # we convert each tensor to a Python list and then JSON-encode it.
        record = {
            "channel": json.dumps(ch.tolist()),              # (n_rx, n_tx) complex
            "bf_matrix": json.dumps(bf_mat.tolist()),        # (n_tx, ?)
            "power_allocation": json.dumps(p_alloc.tolist()),# (n_streams,) or (n_tx,)
            "rate": rate
        }
        data_records.append(record)

    # -----------------------------
    # 4. Convert records to a DataFrame and save to CSV
    # -----------------------------
    df = pd.DataFrame(data_records)

    # Save CSV to the same directory as this script
    output_path = os.path.join(os.path.dirname(__file__), "synthesized_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
