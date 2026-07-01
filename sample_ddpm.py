import argparse
import os
import torch
from torchvision.utils import save_image
from PIL import Image

from models.unet import SimpleUNet
from utils.diffusion import GaussianDiffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=64)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get('args', {})
    out_dir = args.out_dir or train_args.get('out_dir', 'outputs/ddpm_mnist')
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, 'generated_images')
    os.makedirs(image_dir, exist_ok=True)

    model = SimpleUNet(base_channels=train_args.get('base_channels', 64)).to(device)
    model.load_state_dict(ckpt['model'])
    diffusion = GaussianDiffusion(train_args.get('timesteps', 1000), device=device)

    samples = diffusion.sample(model, n=args.num_samples, image_size=28, channels=1)
    save_image((samples + 1) / 2, os.path.join(out_dir, 'generated_grid.png'), nrow=8)

    imgs = ((samples + 1) / 2).clamp(0, 1).cpu()
    for i, img in enumerate(imgs):
        arr = (img.squeeze(0).numpy() * 255).astype('uint8')
        Image.fromarray(arr, mode='L').save(os.path.join(image_dir, f'{i:04d}.png'))
    print(f'Generated images saved to: {out_dir}')


if __name__ == '__main__':
    main()
