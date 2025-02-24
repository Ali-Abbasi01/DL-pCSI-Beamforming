import numpy as np
import torch

class fixed_channel_pga():

    def __init__(self, H, PT):
        self.H = H.numpy()
        self.PT = PT
        
    def solve(self, max_iter=100000000, alpha=0.001, tol=1e-10):
        def logdet(matrix):
            sign, val = np.linalg.slogdet(matrix) / np.log(2)
            return val  # sign should be +1 if matrix is PSD and well-conditioned

        def project_psd_trace(S, PT):
            # Symmetrize for numerical stability
            Ssym = 0.5*(S + S.conj().T)
            
            # Eigen-decomposition
            eigvals, U = np.linalg.eigh(Ssym)
            
            # Clamp negatives to zero
            eigvals[eigvals < 0] = 0
            
            # Scale if trace > PT
            trace_val = np.sum(eigvals)
            if trace_val > PT:
                eigvals = eigvals * (PT / trace_val)

            # Reconstruct
            return (U * eigvals) @ U.conj().T

        m, n = self.H.shape
        # Initialize Sigma
        Sigma = (self.PT/n) * np.eye(n, dtype=complex)

        I_m = np.eye(m, dtype=complex)
        
        for _ in range(max_iter):
            M = I_m + self.H @ Sigma @ self.H.conj().T
            # Gradient wrt Sigma
            M_inv = np.linalg.inv(M)
            G = self.H.conj().T @ M_inv @ self.H
            
            # Take gradient ascent step
            Sigma_new = Sigma + alpha * G
            
            # Project to feasible set
            Sigma_proj = project_psd_trace(Sigma_new, self.PT)
            
            # Check for convergence
            if np.linalg.norm(Sigma_proj - Sigma, 'fro') < tol:
                return Sigma_proj, (logdet(np.eye(m) + self.H @ Sigma_proj @ self.H.conj().T)/np.log(2))
            
            Sigma = Sigma_proj
        
        return Sigma, (logdet(np.eye(m) + self.H @ Sigma @ self.H.conj().T)/np.log(2))


class UIU_pga():

    def __init__(self):

    def solve(self):
