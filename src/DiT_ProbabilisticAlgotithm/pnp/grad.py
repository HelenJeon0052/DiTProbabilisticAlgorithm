import torch
import torch.nn.functional as F



def grad2d_rgb(x):
    # x : [B, 3, L, W]
    """dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1))"""
    
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    
    f = torch.cat([dx, dy], dim=1)
    
    print(f'grad: {f}')

    return f

def div2d_rgb(g):
    # g : [B, 6, L, W]
    dx, dy = g[:, :3], g[:, 3:]
    divx = dx.clone()
    divx[..., :, :-1] -= dx[..., :, 1:]

    divy = dy.clone()
    divy[..., :-1, :] -= dy[..., 1:, :]

    return divx + divy # [B, 3, L, W]
