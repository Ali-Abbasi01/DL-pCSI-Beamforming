import cvxpy as cp
import numpy as np
import torch

class fixed_channel_cvx():

  def __init__(self, H, PT, solver=cp.SCS):
      self.H = H.numpy()
      self.PT = PT
      self.solver = solver

  def solve(self):
      m, n = self.H.shape
      Sigma = cp.Variable((n, n), hermitian=True)
      obj = (cp.log_det(np.eye(m) + self.H @ Sigma @ self.H.conj().T)) / np.log(2.0)
      constraints = [Sigma >> 0, cp.trace(Sigma) <= self.PT]  
      prob = cp.Problem(cp.Maximize(obj), constraints)
      if self.solver == cp.SCS:
          prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-7, verbose=False)
      else:
          prob.solve(solver=solver, verbose=False)
      return Sigma.value, prob.value, prob.status


class UIU_cvx():

  def __init__(self, chan_gen, num_samples=10000):
      self.chan_gen = chan_gen
      self.n_R, self.n_T = chan_gen.generate().shape
      self.num_samples = num_samples
      self.H_bar = chan_gen.H_bar

  def solve(self, PT, solver=cp.SCS):
      H_list = []
      for _ in range(self.num_samples):
          H_list.append(self.chan_gen.generate().numpy())
      H_array = np.array(H_list)
      K, m, n = H_array.shape

      # Define the PSD variable Sigma
      Sigma = cp.Variable((n, n), hermitian=True)

      # Build up sum of log_det(...) over the K realizations
      sum_logdet = 0
      I_m = np.eye(m)
      for k in range(K):
          Hk = H_array[k]
          # log_det(I_m + H_k * Sigma * H_k^H)
          sum_logdet += cp.log_det(I_m + Hk @ Sigma @ Hk.conj().T)

      objective = (1.0 / K) * sum_logdet / np.log(2.0)

      # Constraints: Sigma >= 0 and trace(Sigma) <= P_T
      constraints = [Sigma >> 0, cp.trace(Sigma) <= PT]

      # Set up problem and solve
      prob = cp.Problem(cp.Maximize(objective), constraints)

      prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-7, verbose=False)

      if solver == cp.SCS:
          # Adjust iterations/tolerance for SCS as needed
          prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-7, verbose=False)
      else:
          prob.solve(solver=solver, verbose=False)

      return Sigma.value, prob.value, prob.status
