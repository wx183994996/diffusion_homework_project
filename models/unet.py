import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        c = base_channels
        self.init = nn.Conv2d(in_channels, c, 3, padding=1)

        self.down1 = ResBlock(c, c, time_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(c * 2, c * 2, time_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)

        self.mid1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResBlock(c * 4, c * 4, time_dim)

        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(c * 4, c * 2, time_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.up1 = ResBlock(c * 2, c, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x0 = self.init(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.downsample1(x1)
        x2 = self.down2(x2, t_emb)
        x3 = self.downsample2(x2)
        x3 = self.mid1(x3, t_emb)
        x3 = self.mid2(x3, t_emb)
        y = self.upsample2(x3)
        y = self.up2(torch.cat([y, x2], dim=1), t_emb)
        y = self.upsample1(y)
        y = self.up1(torch.cat([y, x1], dim=1), t_emb)
        return self.out(y)
