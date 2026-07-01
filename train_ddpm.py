import argparse
import csv
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from models.unet import SimpleUNet
from utils.data import get_mnist_loader
from utils.diffusion import GaussianDiffusion
from utils.plot import plot_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='outputs/ddpm_mnist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--sample-every', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    loader = get_mnist_loader(args.data_root, args.batch_size, train=True, num_workers=args.num_workers)
    model = SimpleUNet(base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    csv_path = os.path.join(args.out_dir, 'train_log.csv')
    best_loss = float('inf')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}')
            for x, _ in pbar:
                x = x.to(device)
                loss = diffusion.training_loss(model, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')
            avg_loss = total / len(loader)
            print(f'Epoch {epoch}: avg_loss={avg_loss:.6f}')
            writer.writerow({'epoch': epoch, 'loss': avg_loss})
            f.flush()

            ckpt = {
                'model': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'loss': avg_loss,
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'ckpt_last.pt'))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(args.out_dir, 'ckpt_best.pt'))

            if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
                samples = diffusion.sample(model, n=36, image_size=28, channels=1)
                save_image((samples + 1) / 2, os.path.join(args.out_dir, f'samples_epoch_{epoch:03d}.png'), nrow=6)

    plot_loss(csv_path, os.path.join(args.out_dir, 'loss_curve.png'))
    print(f'Done. Outputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
