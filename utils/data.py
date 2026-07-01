from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(root='./data', batch_size=128, train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True, drop_last=train)
