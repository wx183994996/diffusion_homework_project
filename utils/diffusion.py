from typing import Optional

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x_shape):
        out = values.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def training_loss(self, model, x0: torch.Tensor):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, t: torch.Tensor):
        beta_t = self._extract(self.betas, t, xt.shape)
        alpha_t = self._extract(self.alphas, t, xt.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, xt.shape)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        pred_noise = model(xt, t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, model, n: int, image_size: int = 28, channels: int = 1):
        model.eval()
        x = torch.randn(n, channels, image_size, image_size, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)
