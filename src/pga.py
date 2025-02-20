import numpy as np

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

def mimo_capacity_gradient_ascent(H, PT, max_iter=100000000, alpha=0.001, tol=1e-10):
    """
    Solves the MIMO capacity covariance maximization:
      maximize   log det(I + H*Sigma*H^H)
      s.t.       Sigma >= 0,  trace(Sigma) <= PT
    by projected gradient ascent.
    """
    m, n = H.shape
    # Initialize Sigma
    Sigma = (PT/n) * np.eye(n, dtype=complex)

    I_m = np.eye(m, dtype=complex)
    
    for _ in range(max_iter):
        M = I_m + H @ Sigma @ H.conj().T
        # Gradient wrt Sigma
        M_inv = np.linalg.inv(M)
        G = H.conj().T @ M_inv @ H
        
        # Take gradient ascent step
        Sigma_new = Sigma + alpha * G
        
        # Project to feasible set
        Sigma_proj = project_psd_trace(Sigma_new, PT)
        
        # Check for convergence
        if np.linalg.norm(Sigma_proj - Sigma, 'fro') < tol:
            return Sigma_proj
        
        Sigma = Sigma_proj
    
    return Sigma


# Suppose H is an (m x n) numpy array (complex or real)
m, n = 4, 4
# H = H.numpy()
# H = np.random.randn(m, n) + 1j*np.random.randn(m, n)  # random complex channel
PT = 10.0  # total power
Sigma_opt = mimo_capacity_gradient_ascent(H, PT)
print("Optimal Sigma found by gradient ascent:\n", Sigma_opt)
print("Trace: ", np.trace(Sigma_opt))
print("Objective value = ", logdet(np.eye(m) + H @ Sigma_opt @ H.conj().T))
