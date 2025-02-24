import cvxpy as cp
import numpy as np
import torch

class fixed_channel():

  def __init__(H, PT, solver=cp.SCS):
    self.H = H
    self.PT = PT
    self.soolver = solver

  def solve():
    m, n = self.H.shape
    Sigma = cp.Variable((n, n), hermitian=True)
    obj = (cp.log_det(np.eye(m) + self.H @ Sigma @ self.H.conj().T)) / np.log(2.0)
    constraints = [Sigma >> 0, cp.trace(Sigma) <= self.PT]  
    prob = cp.Problem(cp.Maximize(obj), constraints)
    if solver == cp.SCS:
        prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-7, verbose=False)
    else:
        prob.solve(solver=solver, verbose=False)
    return Sigma.value, prob.value, prob.status


class UIU_channel():

  def __init__(chan_gen, num_samples=10000):
    self.chan_gen = chan_gen
    self.n_R, self.n_T = chan_gen.generate().shape
    self.num_samples = num_samples
    self.H_bar = chan_gen.H_bar
    if torch.all(self.H_bar == 0):
        self.V = chan_gen.U_T
    else:
        U, S, VT = torch.linalg.svd(self.H_bar.T@self.H_bar)
        self.V = U
    self.P = torch.eye(self.n_T)
    self.SNR = SNR

  def solve():
    


def solve_mimo_capacity_cvxpy(H, PT, solver=cp.SCS):
    """
    Solve:
      maximize  log_det(I + H*Sigma*H^H)
      subject to  Sigma >= 0,  trace(Sigma) <= PT
    using CVXPY, with a user-chosen solver.
    """
    m, n = H.shape
    
    # Sigma: n x n, Hermitian
    Sigma = cp.Variable((n, n), hermitian=True)
    
    # log_det(I + H*Sigma*H^H)
    obj = (cp.log_det(np.eye(m) + H @ Sigma @ H.conj().T)) / np.log(2.0)
    
    constraints = [Sigma >> 0, cp.trace(Sigma) <= PT]
    
    prob = cp.Problem(cp.Maximize(obj), constraints)
    
    # Increase iterations and tighten tolerance if using SCS
    if solver == cp.SCS:
        prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-7, verbose=False)
    else:
        prob.solve(solver=solver, verbose=False)

    return Sigma.value, prob.value, prob.status

# Test
if __name__ == "__main__":
    m, n = 4, 4
    H = np.random.randn(m, n) + 1j*np.random.randn(m, n)
    PT = 10.0

    Sigma_opt_cvx, val_cvx, status = solve_mimo_capacity_cvxpy(H, 10)
    
    print("Solver status:", status)
    print("CVX Optimal objective:", val_cvx)
    print("Trace of Sigma from CVX:", np.trace(Sigma_opt_cvx))
    # Compare to water-filling objective or your own calculation
