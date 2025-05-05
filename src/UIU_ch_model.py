# UIU channel model

import torch

class UIU():

    def __init__(self, H_bar, U_R, U_T, G):
        self.H_bar = H_bar
        self.U_R = U_R
        self.U_T = U_T
        self.G = G

    def generate(self):
        n_R, n_T = self.H_bar.shape
        H_tilda = torch.randn(n_R, n_T)
        H = self.H_bar + self.U_R @ (H_tilda * torch.square(self.G)) @ (self.U_T.conj().T)
        return H