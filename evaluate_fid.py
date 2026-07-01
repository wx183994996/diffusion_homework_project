import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from models.unet import SimpleUNet
from utils.data import get_mnist_loader
from utils.diffusion import GaussianDiffusion
from utils.fid import SmallFeatureNet, extract_features, calculate_fid_from_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-root', type=str, default='./data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get('args', {})
    out_dir = train_args.get('out_dir', 'outputs/ddpm_mnist')
    os.makedirs(out_dir, exist_ok=True)

    gen_model = SimpleUNet(base_channels=train_args.get('base_channels', 64)).to(device)
    gen_model.load_state_dict(ckpt['model'])
    diffusion = GaussianDiffusion(train_args.get('timesteps', 1000), device=device)
    feat_model = SmallFeatureNet().to(device).eval()

    real_loader = get_mnist_loader(args.data_root, args.batch_size, train=False, num_workers=2)
    real_feats = []
    seen = 0
    for x, _ in tqdm(real_loader, desc='Extract real features'):
        take = min(x.shape[0], args.num_samples - seen)
        if take <= 0:
            break
        real_feats.append(extract_features(feat_model, x[:take], device))
        seen += take
    real_feats = np.concatenate(real_feats, axis=0)

    fake_feats = []
    remaining = args.num_samples
    while remaining > 0:
        n = min(args.batch_size, remaining)
        samples = diffusion.sample(gen_model, n=n, image_size=28, channels=1)
        fake_feats.append(extract_features(feat_model, samples, device))
        remaining -= n
    fake_feats = np.concatenate(fake_feats, axis=0)

    fid = calculate_fid_from_features(real_feats, fake_feats)
    report = os.path.join(out_dir, 'fid_report.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('DDPM MNIST FID-like Evaluation Report\n')
        f.write('====================================\n')
        f.write(f'Checkpoint: {args.ckpt}\n')
        f.write(f'Num samples: {args.num_samples}\n')
        f.write(f'FID-like score: {fid:.6f}\n')
        f.write('\nNote: This project uses a lightweight CNN feature extractor for classroom evaluation.\n')
        f.write('It is suitable for relative comparison in this assignment, but is not standard ImageNet Inception FID.\n')
    print(f'FID-like score: {fid:.6f}')
    print(f'Report saved to: {report}')


if __name__ == '__main__':
    main()
