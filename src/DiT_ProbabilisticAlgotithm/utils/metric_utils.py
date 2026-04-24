import torch




def normalize_sigma(sigma, g):
    if not torch.is_tensor(sigma):
      sigma = torch.tensor(sigma, device=g.device, dtype=g.dtype)
    if sigma.ndim == 0:
      sigma = sigma[None]
    if sigma.shape[0] == 1 and g.shape[0] > 1:
      sigma = sigma.repeat(g.shape[0])
    if sigma.ndim == 2 and sigma.shape[1] == 1:
      sigma = sigma[:, 0]
    return sigma
    