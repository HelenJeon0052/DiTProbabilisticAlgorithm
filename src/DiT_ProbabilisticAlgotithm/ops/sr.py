import torch





class DownsampleA:
    def __init(self, scale=2):
        self.s = scale

    def A(self, x):
        # x : [B, 3, L, W] > [B, 3, L/s, W/s]
        return x[..., ::self.s, ::self.s]

    def AT(self, y, out_lw):
        # y : [B, 3, l, w] > [B, 3, L, W]

        B, C, l, w = y.shape
        L, W = out_lw
        x = torch.zeros((B, C, L, W), device=y.device, dtype=y.dtype)
        x[..., ::self.s, ::self.s] = y
        return x