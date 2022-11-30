import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to training curves data in .json format",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Draw metrics
    plt.figure(figsize=(12.8, 7.2), dpi=300)
    for metrics in ('rouge-1', 'rouge-2', 'rouge-L'):
        with open(os.path.join(args.path, f"{metrics}.json"), "r") as f:
            arr = np.array([x[1:] for x in json.load(f)])
            x, y = arr[:, 0], arr[:, 1]
            plt.plot(x, y, label=metrics)

    plt.xlabel('Steps')
    plt.ylabel('Score (x100)')
    plt.legend()
    plt.savefig(os.path.join(args.path, 'metrics.png'))
    plt.clf()

    # Draw training loss
    plt.figure(figsize=(12.8, 7.2), dpi=300)
    with open(os.path.join(args.path, f"train-loss.json"), "r") as f:
        arr = np.array([x[1:] for x in json.load(f)])
        x, y = arr[:, 0], arr[:, 1]
        plt.plot(x, y, label='Train-loss', linewidth=0.25)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.path, 'train-loss.png'))
    plt.clf()


if __name__ == "__main__":
    main()
