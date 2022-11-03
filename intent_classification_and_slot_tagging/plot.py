import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use('AGG')

from matplotlib import pyplot as plt

experiments = list(map(lambda x: Path(x), [
    "./ckpt/intent.exp001",
    "./ckpt/intent.exp002",
    "./ckpt/intent.exp003",
    "./ckpt/intent.exp004",
    "./ckpt/intent.exp005",
    "./ckpt/intent.exp006",
]))

def main():
    metrics: List[Dict] = []
    params: List[Dict] = []

    for exp in experiments:
        with open(exp / "metrics.json", "r") as f:
            metrics.append(json.load(f))

        with open(exp / "param.json", "r") as f:
            params.append(json.load(f))

    labels = [
        "hidden_size={}, {}-directional".format(
            param['hidden_size'], 'bi' if param['bidirectional'] else 'uni'
        ) for param in params
    ]
    epochs = [ metric['epochs'] for metric in metrics ][0]
    loss = [ metric['val_loss'] for metric in metrics ]
    acc = [ metric['val_acc'] for metric in metrics ]

    _, axs = plt.subplots(1, 2, figsize=(12.8, 7.2))
    axs[0].set_title('Validation loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 3)
    axs[1].set_title('Validation accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_ylim(0)
    for l, a, label in zip(loss, acc, labels):
        axs[0].plot(epochs, l, label=label)
        axs[1].plot(epochs, a, label=label)
    axs[0].legend()
    axs[1].legend()

    plt.savefig('metrics.png')

    return

if __name__ == "__main__":
    main()
