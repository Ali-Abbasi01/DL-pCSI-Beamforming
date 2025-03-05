#Algorithm
import torch

class UIU_algorithm():

    def __init__(self, chan_gen, SNR, num_samples=10000):
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

    def H_gen(self):
        self.H = self.chan_gen.generate()
        return self.H

    def H_hat(self): return self.H @ self.V

    def h_hat(self, j): return self.H_hat()[:, j].reshape(-1, 1)

    def B(self, j):
        H_j_hat = torch.cat((self.H_hat()[:, :j], self.H_hat()[:, j+1:]), dim=1)
        P_new = torch.cat((self.P[:j, :], self.P[j+1:, :]), dim=0)  # Remove j-th row
        P_j = torch.cat((P_new[:, :j], P_new[:, j+1:]), dim=1)
        return torch.linalg.inv(torch.eye(self.n_R) + (self.SNR / self.n_T) * (H_j_hat @ P_j @ H_j_hat.T))
    
    def MMSE(self, j): return 1/(1 + self.P[j, j]*(self.SNR/self.n_T)*((self.h_hat(j).T)@self.B(j)@self.h_hat(j)))[0, 0]

    def MMSE_bar(self, j):
        s = 0
        for i in range(self.num_samples):
            self.H_gen()
            s += self.MMSE(j)
        return s/self.num_samples
    
    def Eq12(self, j):
        s = 0
        for i in range(self.num_samples):
            self.H_gen()
            s += (self.h_hat(j).T)@self.B(j)@self.h_hat(j)
        return (self.SNR*s)/(self.n_T*self.num_samples)

    def alg(self):
        while True: 
            flag = 0
            #Allocation
            for k in range(8):
                diag = []
                den = (sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T)
                for j in range(self.n_T):
                    diag.append((1 - self.MMSE_bar(j))/den)
                torch.diagonal(self.P).copy_(torch.tensor(diag))
            #Check
            rhs = sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T
            for i in range(self.n_T):
                if abs(self.P[i, i]) <= 0.001 * self.n_T:
                    if self.Eq12(i) > rhs:
                        l = [self.Eq12(k) for k in range(self.n_T)]
                        idx = l.index(min(l))
                        self.P[idx, idx] = 0
                        flag = 1
                        break  
            if flag == 0: break
        return self.V, self.P

class wf_algorithm(): 

    def __init__(self, H, Pt):
        self.H = H
        self.Pt = Pt
        self.U, self.S, self.V = torch.svd(self.H)

    def bf_matrix(self):
        return self.V

    def p_allocation(self):
        delta = 1/(self.S**2)
        # self.S = torch.cat((self.S, torch.zeros(self.V.shape[0] - self.S.shape[0], dtype=S.dtype)), dim=0)
        def f(wl):
            s = 0
            for i in delta:
                s += max(0, wl-i)
            return s
        wll = torch.arange(torch.min(delta), self.Pt+torch.max(delta)+1, 0.01)
        p = torch.tensor([f(i) for i in wll])
        wl_star = wll[torch.argmin(torch.abs(p - self.Pt))]
        diags = torch.zeros(self.S.shape[0], dtype=torch.complex64)
        for i in range(self.S.shape[0]):
            diags[i] = max(0, wl_star-(1/(self.S[i]**2)))
        # diags = [max(0, wl_star-delta[0]), max(0, wl_star-delta[1]),max(0, wl_star-delta[2]),0,0]
        D = torch.diag(diags)
        return D               
              
