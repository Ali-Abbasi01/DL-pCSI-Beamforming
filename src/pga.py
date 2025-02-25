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

    def __init__(self, chan_gen, num_samples=10000):
        self.chan_gen = chan_gen
        self.num_samples = num_samples
        
    def solve(self, PT, max_iter=10000, alpha=1e-3, tol=1e-8):
        def project_psd_trace(S, PT):
            """Project S onto the set {S >= 0, tr(S) <= PT}."""
            # Make Hermitian for numerical stability
            Ssym = 0.5 * (S + S.conj().T)

            # Eigen-decompose
            eigvals, U = np.linalg.eigh(Ssym)

            # Zero out negative eigenvalues (PSD constraint)
            eigvals[eigvals < 0] = 0

            # Scale down if trace > PT
            tr_val = np.sum(eigvals)
            if tr_val > PT:
                eigvals *= (PT / tr_val)

            # Reconstruct
            return (U * eigvals) @ U.conj().T

        def logdet_psd(A):
            """Returns log2(det(A)), assuming A is PSD and well-conditioned."""
            # np.linalg.slogdet returns (sign, log_abs_det) in natural log
            sign, ld = np.linalg.slogdet(A)
            if sign <= 0:
                return -np.inf  # or handle numerically invalid matrix
            return ld / np.log(2.0)

        H_list = []
        for _ in range(self.num_samples):
            H_list.append(self.chan_gen.generate().numpy())
        H_array = np.array(H_list)
        K, m, n = H_array.shape
        I_m = np.eye(m, dtype=complex)

        # Initialize Sigma (PSD) with uniform power across n dimensions
        Sigma = (PT / n) * np.eye(n, dtype=complex)

        for it in range(max_iter):
            # 1) Compute gradient wrt Sigma as average over all channels
            grad = np.zeros_like(Sigma, dtype=complex)
            
            for k in range(K):
                Hk = H_list[k]  # shape (m, n)
                M_k = I_m + Hk @ Sigma @ Hk.conj().T  # shape (m, m)
                M_k_inv = np.linalg.inv(M_k)  # (m, m)

                # Derivative: d/dSigma [log det(M_k)] = H_k^H * M_k_inv * H_k
                grad_k = Hk.conj().T @ M_k_inv @ Hk
                grad += grad_k  # sum across channels

            grad /= K  # average the gradient

            # 2) Gradient ascent step
            Sigma_new = Sigma + alpha * grad

            # 3) Project back onto {Sigma >= 0, trace(Sigma) <= P_T}
            Sigma_proj = project_psd_trace(Sigma_new, PT)

            # 4) Check convergence
            diff_norm = np.linalg.norm(Sigma_proj - Sigma, 'fro')
            if diff_norm < tol:
                Sigma = Sigma_proj
                break

            Sigma = Sigma_proj

        # Compute final approximate objective value
        # i.e. the average of log2(det(...)) across channels
        avg_logdet = 0.0
        for k in range(K):
            Hk = H_list[k]
            Mk = I_m + Hk @ Sigma @ Hk.conj().T
            avg_logdet += logdet_psd(Mk)
        avg_logdet /= K  # average in bits

        return Sigma, avg_logdet
