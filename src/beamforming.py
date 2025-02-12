#Algorithm
class algorithm():

    def __init__(self, num_samples=10000, chan_gen, SNR):
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
        H_j_hat = torch.cat((self.H[:, :j], self.H[:, j+1:]), dim=1)
        P_new = torch.cat((self.P[:j, :], self.P[j+1:, :]), dim=0)  # Remove j-th row
        P_j = torch.cat((P_new[:, :j], P_new[:, j+1:]), dim=1)
        return torch.linalg.inv(torch.eye(self.n_R) + (self.SNR / self.n_T) * (H_j_hat @ P_j @ H_j_hat.T))
    
    def MMSE(self, j): return 1/(1 + self.P[j, j]*(self.SNR/self.n_T)*((self.h_hat(j).T)@self.B(j)@self.h_hat(j)))

    def MMSE_bar(self, j):
        s = 0
        for i in range(self.num_samples):
            self.H_gen()
            s += self.MMSE(j)
        return s/self.num_samples
    
    def P1(self):
        for k in range(10):
            ss = (sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T)
            p0 = (1 - self.MMSE_bar(0))/ss
            p1 = (1 - self.MMSE_bar(1))/ss
            p2 = (1 - self.MMSE_bar(2))/ss
            print(p0, p1, p2)
            print('\n\n')
            self.P[0, 0] = p0
            self.P[1, 1] = p1
            self.P[2, 2] = p2
    
    def Eq12(self, j):
        s = 0
        for i in range(self.num_samples):
            self.H_gen()
            s += (self.h_hat(j).T)@self.B(j)@self.h_hat(j)
            return (self.SNR*s)/(self.n_T*self.num_samples)

    
    def alg(self):
        while True:
            flag = 0
            diag = []
            ss = (sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T)
            for j in range(self.n_T):
                diag.append((1 - self.MMSE_bar(j))/ss)
            np.fill_diagonal(self.P, diag)
            print(self.P)

            for j in range(self.n_T):
                if self.P[j, j] == 0:
                    if self.Eq12(j) <= sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T: pass
                    else:
                        l = [self.Eq12(k) for k in range(self.n_T)]
                        idx = l.index(min(l))
                        self.P[idx, idx] = 0
                        flag = 1
                        break
                else:
                    if self.Eq12(j) > sum([(1 - self.MMSE_bar(i)) for i in range(self.n_T)])/self.n_T: pass
                    else:
                        l = [self.Eq12(k) for k in range(self.n_T)]
                        idx = l.index(min(l))
                        self.P[idx, idx] = 0 
                        flag = 1  
                        break
            print(self.P)
            if flag == 1:
                continue  
            break
        return self.P