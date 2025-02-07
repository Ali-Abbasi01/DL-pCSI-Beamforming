#Algorithm
class algorithm():

    def __init__(self, num_samples=10000):
        self.n_T = 3
        self.n_R = 2
        self.num_samples = num_samples
        H_bar = np.ones((2, 3))
        U, S, VT = np.linalg.svd(H_bar.T@H_bar)
        self.V = U
        self.P = np.identity(3)
        self.SNR = 3.16

    def H_gen(self):
        H_bar = np.ones((2, 3))
        H_tilda = np.random.normal(0, 1, size=(2, 3))
        self.H = (1/np.sqrt(2))*(H_bar + H_tilda)
        return self.H

    def H_hat(self): return self.H @ self.V

    def h_hat(self, j): return self.H_hat()[:, j].reshape(-1, 1)

    def B(self, j):
        H_j_hat = np.delete(self.H_hat(), j, axis=1)
        P_j = np.delete(np.delete(self.P, j, axis=0), j, axis=1)
        return np.linalg.inv(np.identity(self.n_R) + (self.SNR / self.n_T) * (H_j_hat @ P_j @ H_j_hat.conj().T))
    
    def MMSE(self, j): return np.real((1/(1 + self.P[j, j]*(self.SNR/self.n_T)*((self.h_hat(j).conj().T)@self.B(j)@self.h_hat(j))))[0, 0])

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
            s += (self.h_hat(j).conj().T)@self.B(j)@self.h_hat(j)
            return np.real(((self.SNR*s)/(self.n_T*self.num_samples))[0, 0])

    
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