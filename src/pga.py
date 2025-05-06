import numpy as np
import torch
class fixed_channel_pga():

    def __init__(self, H, PT):
        self.H = H
        self.PT = PT
        self.Nr, self.Nt = H.shape
        
    def solve(self, num_iter=200000, lr=0.01):
        def proj_psd_trace(S, P):
            """Project Hermitian S onto {X ≽ 0,  tr(X) ≤ P}."""
            # Hermitian eigendecomp
            eigval, eigvec = torch.linalg.eigh(S)
            eigval.clamp_(min=0)             # PSD
            s = eigval.sum()
            if s > P:                        # scale down uniformly
                eigval *= P / s
            return (eigvec * eigval) @ eigvec.conj().T
        
        I = torch.eye(self.Nr, dtype=torch.cfloat)

        Sigma = torch.eye(self.Nt, dtype=torch.cfloat, requires_grad=True)

        for _ in range(num_iter): 
            Sigma.requires_grad_(True)
            M = I + self.H @ Sigma @ self.H.conj().T
            loss = torch.logdet(M).real
            loss.backward()
            g = Sigma.grad

            with torch.no_grad():
                Sigma = Sigma + lr * g
                Sigma = proj_psd_trace(Sigma, self.PT)

        return Sigma
    
class Bnetwork_channel_pga():

    def __init__(self, P, PT):
        self.p = P
        self.PT = PT
        Br = self.p.calculate_Br()
        Bt = self.p.calculate_Bt()
        A = self.p.calculate_A()
        H = Br @ A @ Bt.conj().T
        self.Nr, self.Nt = H.shape
        
    def solve(self, num_iter=100000, num_sto=1000, lr=0.01):
        def proj_psd_trace(S, P):
            """Project Hermitian S onto {X ≽ 0,  tr(X) ≤ P}."""
            # Hermitian eigendecomp
            eigval, eigvec = torch.linalg.eigh(S)
            eigval.clamp_(min=0)             # PSD
            s = eigval.sum()
            if s > P:                        # scale down uniformly
                eigval *= P / s
            return (eigvec * eigval) @ eigvec.conj().T

        def obj_sg(Sigma):
            loss = 0
            for i in range(num_sto):
                Br = self.p.calculate_Br()
                Bt = self.p.calculate_Bt()
                A = self.p.calculate_A()
                H = Br @ A @ Bt.conj().T
                M = I + H @ Sigma @ H.conj().T
                loss += torch.logdet(M)
            return loss/num_sto
        
        I = torch.eye(self.Nr, dtype=torch.cfloat)

        Sigma = torch.eye(self.Nt, dtype=torch.cfloat, requires_grad=True)

        for _ in range(num_iter): 
            Sigma.requires_grad_(True)
            loss = obj_sg(Sigma).real
            loss.backward()
            g = Sigma.grad

            with torch.no_grad():
                Sigma = Sigma + lr * g
                Sigma = proj_psd_trace(Sigma, PT)

        return Sigma

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
