import os
import sys
import json
import torch
import pandas as pd
import importlib
# Get the current working directory
scripts_dir = os.getcwd()
# Go up two levels
project_root = os.path.abspath(os.path.join(scripts_dir, '..', '..'))
sys.path.append(project_root)

import src.beamforming
importlib.reload(src.beamforming)
from src.beamforming import wf_algorithm
import src.utils
importlib.reload(src.utils)
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

def main(num_samples, n_T, n_R, Pt):

    num_samples = num_samples
    n_rx = n_R 
    n_tx = n_T
    Pt = Pt

    channels = generate_random_mimo_channels(num_samples, n_rx, n_tx)

    data_records = []

    for i in range(num_samples):
        ch = channels[i] 

        wf = wf_algorithm(ch, Pt) 

        bf_mat = wf.bf_matrix()
        p_alloc = wf.p_allocation()
        Cov = bf_mat @ p_alloc @ bf_mat.conj().T
        rate = calculate_rate(ch, Cov).real
        # If `rate` is a torch scalar, convert to Python float
        if isinstance(rate, torch.Tensor):
            rate = rate.item()


        record = {
            "channel": json.dumps({"real": ch.real.tolist(), "imag": ch.imag.tolist()}),  # Convert complex to JSON
            "bf_matrix": json.dumps({"real": bf_mat.real.tolist(), "imag": bf_mat.imag.tolist()}), 
            "p_allocation": json.dumps({"real": p_alloc.real.tolist(), "imag": p_alloc.imag.tolist()}),
            "rate": rate 
        }
        data_records.append(record)

    df = pd.DataFrame(data_records)

    # Save CSV to the same directory as this script
    output_path = os.path.join(os.getcwd(), "synthesized_data_fixed.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

num_samples = 1000
n_T = 4
n_R = 4
Pt = 4

if __name__ == "__main__":
    main(num_samples, n_T, n_R, Pt)
 