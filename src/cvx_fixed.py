import cvxpy as cp
import numpy as np

def solve_mimo_capacity_cvxpy(H, PT):
    """
    Solve:
      maximize  log_det(I + H * Sigma * H^H)
      subject to  Sigma >= 0,  trace(Sigma) <= PT
    using CVXPY.
    """
    m, n = H.shape
    
    # Declare Sigma as Hermitian (complex PSD).
    # 'hermitian=True' ensures Sigma == Sigma.H.
    Sigma = cp.Variable((n, n), hermitian=True)

    # Objective: log_det(I + H*Sigma*H^H).
    # If Sigma is Hermitian and PSD, then (I + H*Sigma*H^H) will also be Hermitian PSD.
    expr = cp.log_det(np.eye(m) + H @ Sigma @ H.conj().T)

    # Constraints: PSD and trace <= PT
    constraints = [Sigma >> 0, cp.trace(Sigma) <= PT]

    prob = cp.Problem(cp.Maximize(expr), constraints)
    prob.solve(solver=cp.SCS, verbose=False)  # or another solver like 'MOSEK'
    
    return Sigma.value, prob.value

# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    m, n = 4, 4
    H = np.random.randn(m, n) + 1j*np.random.randn(m, n)
    PT = 10.0
    
    Sigma_opt_cvx, val_cvx = solve_mimo_capacity_cvxpy(H, 10)
    print("CVX Optimal objective:", val_cvx)
    print("CVX Sigma:\n", Sigma_opt_cvx)
    print("Trace of Sigma:", np.trace(Sigma_opt_cvx))
