import csv
import matplotlib.pyplot as plt


def plot_loss(csv_path: str, out_path: str):
    epochs, losses = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Noise Prediction Loss')
    plt.title('DDPM Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
