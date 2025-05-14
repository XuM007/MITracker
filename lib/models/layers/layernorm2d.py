import torch
from torch import nn

#----------------- from SAM ----------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if len(x.shape) == 4:
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        elif len(x.shape) == 5:
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x