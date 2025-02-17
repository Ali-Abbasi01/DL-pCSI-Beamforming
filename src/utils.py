def calculate_rate(H, sigma):
    n_r = H.shape[0]
    I = torch.eye(n_r, dtype=H.dtype)
    M = I + H @ sigma @ H.conj().T
    rate = torch.log2(torch.det(M))
    return rate