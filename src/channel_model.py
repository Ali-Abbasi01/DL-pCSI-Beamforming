class beam_netwrok():

    def __init__(self, num_RX_ant, num_TX_ant, num_scatterers, lam, Ant_dist, TX_loc, RX_loc, SC_locs, rand_ph):
        self.num_RX_ant = num_RX_ant
        self.num_TX_ant = num_TX_ant
        self.num_scatterers = num_scatterers
        self.lam = lam
        self.Ant_dist = Ant_dist
        self.TX_loc = TX_loc
        self.RX_loc = RX_loc
        self.SC_locs = SC_locs
        self.rand_ph = rand_ph
        self.T_locs = [[i*self.Ant_dist+self.TX_loc[0], self.TX_loc[1]]  for i in range(self.num_TX_ant)]
        self.R_locs = [[i*self.Ant_dist+self.RX_loc[0], self.RX_loc[1]]  for i in range(self.num_RX_ant)]

    def calculate_Bt(self):
        Bt = torch.zeros(self.num_TX_ant, self.num_scatterers+1, dtype=torch.complex64)
        # qT = (RX_locs[self.RX_idx] - TX_locs[self.TX_idx])
        qT = (self.RX_loc - self.TX_loc)/torch.norm((self.RX_loc - self.TX_loc))
        D = torch.tensor(self.T_locs) - torch.tile(self.TX_loc, (self.num_TX_ant, 1))
        Bt[:, 0] = torch.exp(((-2*torch.pi*1j)/Lam)*(torch.matmul(qT, torch.transpose(D, 0, 1))))
        for i, S_loc in enumerate(self.SC_locs):
            # qT = (S_loc - TX_locs[self.TX_idx])
            qT = (S_loc - self.TX_loc)/torch.norm((S_loc - self.TX_loc))
            Bt[:, i+1] = torch.exp(((-2*torch.pi*1j)/Lam)*(torch.matmul(qT, torch.transpose(D, 0, 1))))
        return Bt

    def calculate_Br(self):
        Br = torch.zeros(self.num_RX_ant, self.num_scatterers+1, dtype=torch.complex64)
        # qR = (RX_locs[self.RX_idx] - TX_locs[self.TX_idx])
        qR = (self.RX_loc - self.TX_loc)/torch.norm((self.RX_loc - self.TX_loc))
        D = torch.tensor(self.R_locs) - torch.tile(self.RX_loc, (self.num_RX_ant, 1))
        Br[:, 0] = torch.exp(((2*torch.pi*1j)/Lam)*(torch.matmul(qR, torch.transpose(D, 0, 1))))
        for i, S_loc in enumerate(self.SC_locs):
            # qR = (S_loc - RX_locs[self.RX_idx])
            qR = (self.RX_loc - S_loc)/torch.norm((S_loc - self.RX_loc))
            Br[:, i+1] = torch.exp(((2*torch.pi*1j)/Lam)*(torch.matmul(qR, torch.transpose(D, 0, 1))))
        return Br

    def calculate_A(self, L = None):
        A = torch.zeros(self.num_scatterers+1, self.num_scatterers+1, dtype=torch.complex64)
        r = torch.norm((self.RX_loc - self.TX_loc))
        A[0, 0] = (torch.exp(-2*torch.pi*(r/Lam)*1j))/r
        if self.rand_ph:
            for i, S_loc in enumerate(self.SC_locs):
                r = torch.norm((S_loc - self.TX_loc)) + torch.norm((self.RX_loc - S_loc))
                random_phase = torch.rand(1) * 2 * torch.pi
                A[i+1, i+1] = ((torch.exp(-2*torch.pi*(r/Lam)*1j))*(torch.exp(random_phase*1j)))/r
        else:
            for i, S_loc in enumerate(self.SC_locs):
                r = torch.norm((S_loc - self.TX_loc)) + torch.norm((self.RX_loc - S_loc))
                # rand_ph = torch.rand(1) * 2 * torch.pi
                # rand_ph = torch.tensor(torch.pi/6)
                phase = L[i]
                A[i+1, i+1] = ((torch.exp(-2*torch.pi*(r/Lam)*1j))*(torch.exp(phase*1j)))/r
        return A

    def generate(self, L = None):
        H = self.calculate_Br() @ self.calculate_A(L) @ (self.calculate_Bt().conj().T)
        return H


class UIU():

    def __init__(self, H_bar, U_R, U_T, G):
        self.H_bar = H_bar
        self.U_R = U_R
        self.U_T = U_T
        self.G = G

    def generate(self):
        n_R, n_T = H_bar.shape
        H_tilda = torch.randn(n_R, n_T)
        H = self.H_bar + self.U_R @ (H_tilda * torch.square(self.G)) @ (self.U_T.conj().T)
        return H