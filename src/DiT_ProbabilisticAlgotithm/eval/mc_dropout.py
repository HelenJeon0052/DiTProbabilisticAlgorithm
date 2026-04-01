import torch
import torch.nn as nn



def enable_mc_dropout(model: nn.Module):
    model.eval()

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


@torch.no_grad()
def mn_dropout_predict(model:nn.Module, x, sigma, K=8):
    enable_mc_dropout(model)

    outs = []

    for _ in range(K):
        outs.append(model(x, sigma))

    outs = torch.stack(outs, dim=0)

    mean = outs.mean(dim=0)
    var = outs.var(dim=0) # epistemic uncertainty proxy

    # per-sample scalar uncertainty
    u = var.mean(dim=tuple(range(1, var.ndim))) # [B]

    return mean, var, u



