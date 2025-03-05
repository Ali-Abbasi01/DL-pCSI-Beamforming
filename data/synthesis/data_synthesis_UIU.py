import os
import sys
import json
import torch
import pandas as pd
import importlib

# Get the current working directory
scripts_dir = os.getcwd()
# Go up two levels to find the project root (adjust as needed for your repo structure)
project_root = os.path.abspath(os.path.join(scripts_dir, '..', '..'))
sys.path.append(project_root)

# Import your custom modules
# (Replace 'src.uiu' and 'src.uiu_algorithm' with the actual module names)
import src.beamforming
importlib.reload(src.beamforming)
from src.beamforming import UIU_algorithm  # example import

import src.channel_model
importlib.reload(src.channel_model)
from src.channel_model import UIU  # example import

def generate_random_unitary(n):
    """
    Generate an n x n random orthonormal matrix by:
    1) Creating a random real matrix with entries ~ N(0, 1).
    2) Performing QR factorization.
    3) Normalizing diagonal of R (optional) to avoid negative scaling.
    """
    # Create a random real matrix
    Z = torch.randn(n, n)

    # QR factorization
    Q, R = torch.qr(Z)

    # Ensure the diagonal of R is positive (this prevents sign flips in Q)
    diag = torch.diagonal(R, 0)
    signs = diag.sign()
    Q = Q * signs

    return Q

def generate_random_uniform_matrix(n_rows, n_cols, a, b, dtype=torch.float32):
    """
    Generates an n_rows x n_cols real matrix with entries drawn uniformly from [a, b].
    """
    return (b - a) * torch.rand(n_rows, n_cols, dtype=dtype) + a

def main(num_samples, n_T, n_R, Pt, a, b):
    """
    Generates #num_samples random sets of (U_R, U_T, H_bar, G) under the UIU model.
    For each set, it runs UIU_algorithm to produce V and P, and saves all data to a CSV.
    """
    # You could treat Pt as an SNR directly or define SNR = Pt explicitly
    SNR = n_T

    data_records = []

    for i in range(num_samples):
        # 1) Generate random U_R and U_T (both unitary)
        U_R = generate_random_unitary(n_R)
        U_T = generate_random_unitary(n_T)

        # 2) Generate H_bar and G (uniform in [a, b])
        H_bar = generate_random_uniform_matrix(n_R, n_T, a, b)
        G = generate_random_uniform_matrix(n_R, n_T, a, b)

        # 3) Create a UIU channel generator instance
        #    Adjust constructor signature as needed
        chan_gen = UIU(H_bar, U_R, U_T, G)

        # 4) Create an algorithm instance and run it
        #    Adjust constructor and method calls as needed for your classes
        algorithm = UIU_algorithm(chan_gen, SNR, num_samples=10000)
        V, P = algorithm.alg()

        # Convert tensors to Python-native structures for CSV
        # If U_R, U_T, V, P are complex, we convert to real/imag JSON
        def real_to_json(mat):
            return json.dumps(mat.cpu().tolist())

        record = {
            "U_R": real_to_json(U_R),
            "U_T": real_to_json(U_T),
            "H_bar": real_to_json(H_bar),  # real matrix
            "G": real_to_json(G),          # real matrix
            "V": real_to_json(V),
            "P": real_to_json(P)
        }
        data_records.append(record)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data_records)
    output_path = os.path.join(os.getcwd(), "synthesized_data_UIU.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    # Adjust 'a' and 'b' to your desired uniform distribution range
    num_samples = 1
    n_T = 4
    n_R = 4
    Pt = 4
    a, b = -1.0, 1.0

    main(num_samples, n_T, n_R, Pt, a, b)
