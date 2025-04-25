import torch

class Network():

    num_RX_ant = None
    num_TX_ant = None
    num_scatterers = None
    lam = None
    Ant_dist = None
    TX_locs = None
    RX_locs = None
    SC_locs = None
    rand_ph = None

class pair():
    def __init__(self, N, RX_idx, TX_idx):
        self.TX_idx = TX_idx
        self.RX_idx = RX_idx
        self.num_RX_ant = N.num_RX_ant
        self.num_TX_ant = N.num_TX_ant
        self.num_scatterers = N.num_scatterers
        self.lam = N.lam
        self.Ant_dist = N.Ant_dist
        self.TX_locs = N.TX_locs
        self.RX_locs = N.RX_locs
        self.SC_locs = N.SC_locs
        self.rand_ph = N.rand_ph
        self.T_locs = [[i*self.Ant_dist+self.TX_locs[self.TX_idx][0], self.TX_locs[self.TX_idx][1]]  for i in range(self.num_TX_ant)]
        self.R_locs = [[i*self.Ant_dist+self.RX_locs[self.RX_idx][0], self.RX_locs[self.RX_idx][1]]  for i in range(self.num_RX_ant)]
        self.S_locs = [i for i in torch.from_numpy(self.SC_locs[self.RX_idx, self.TX_idx])]

    def calculate_Bt(self):
        Bt = torch.zeros(self.num_TX_ant, self.num_scatterers[self.RX_idx, self.TX_idx]+1, dtype=torch.complex64)
        # qT = (RX_locs[self.RX_idx] - TX_locs[self.TX_idx])
        qT = (self.RX_locs[self.RX_idx] - self.TX_locs[self.TX_idx])/torch.norm((self.RX_locs[self.RX_idx] - self.TX_locs[self.TX_idx]))
        D = torch.tensor(self.T_locs) - torch.tile(self.TX_locs[self.TX_idx], (self.num_TX_ant, 1))
        Bt[:, 0] = torch.exp(((-2*torch.pi*1j)/self.lam)*(torch.matmul(qT, torch.transpose(D, 0, 1))))
        for i, S_loc in enumerate(self.S_locs):
            # qT = (S_loc - TX_locs[self.TX_idx])
            qT = (S_loc - self.TX_locs[self.TX_idx])/torch.norm((S_loc - self.TX_locs[self.TX_idx]))
            Bt[:, i+1] = torch.exp(((-2*torch.pi*1j)/self.lam)*(torch.matmul(qT, torch.transpose(D, 0, 1))))
        return Bt

    def calculate_Br(self):
        Br = torch.zeros(self.num_RX_ant, self.num_scatterers[self.RX_idx, self.TX_idx]+1, dtype=torch.complex64)
        # qR = (RX_locs[self.RX_idx] - TX_locs[self.TX_idx])
        qR = (self.RX_locs[self.RX_idx] - self.TX_locs[self.TX_idx])/torch.norm((self.RX_locs[self.RX_idx] - self.TX_locs[self.TX_idx]))
        D = torch.tensor(self.R_locs) - torch.tile(self.RX_locs[self.RX_idx], (self.num_RX_ant, 1))
        Br[:, 0] = torch.exp(((2*torch.pi*1j)/self.lam)*(torch.matmul(qR, torch.transpose(D, 0, 1))))
        for i, S_loc in enumerate(self.S_locs):
            # qR = (S_loc - RX_locs[self.RX_idx])
            qR = (self.RX_locs[self.RX_idx] - S_loc)/torch.norm((S_loc - self.RX_locs[self.RX_idx]))
            Br[:, i+1] = torch.exp(((2*torch.pi*1j)/self.lam)*(torch.matmul(qR, torch.transpose(D, 0, 1))))
        return Br

    def calculate_A(self, L = None):
        A = torch.zeros(len(self.S_locs)+1, len(self.S_locs)+1, dtype=torch.complex64)
        r = torch.norm((self.RX_locs[self.RX_idx] - self.TX_locs[self.TX_idx]))
        A[0, 0] = (torch.exp(-2*torch.pi*(r/self.lam)*1j))/r
        if self.rand_ph:
            for i, S_loc in enumerate(self.S_locs):
                r = torch.norm((S_loc - self.TX_locs[self.TX_idx])) + torch.norm((self.RX_locs[self.RX_idx] - S_loc))
                random_phase = torch.rand(1) * 2 * torch.pi
                A[i+1, i+1] = ((torch.exp(-2*torch.pi*(r/self.lam)*1j))*(torch.exp(random_phase*1j)))/r
        else:
            for i, S_loc in enumerate(self.S_locs):
                r = torch.norm((S_loc - self.TX_locs[self.TX_idx])) + torch.norm((self.RX_locs[self.RX_idx] - S_loc))
                # rand_ph = torch.rand(1) * 2 * torch.pi
                # rand_ph = torch.tensor(torch.pi/6)
                phase = L[i]
                A[i+1, i+1] = ((torch.exp(-2*torch.pi*(r/self.lam)*1j))*(torch.exp(phase*1j)))/r
        return A