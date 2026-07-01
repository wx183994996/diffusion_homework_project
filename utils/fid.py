import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm


class SmallFeatureNet(nn.Module):
    """轻量特征网络。用于课堂作业的 FID-like 统计特征，不等同于标准 Inception FID。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def calculate_fid_from_features(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@torch.no_grad()
def extract_features(model, images, device):
    # images are in [-1, 1], convert to [0, 1]
    x = (images + 1) / 2
    return model(x.to(device)).cpu().numpy()
