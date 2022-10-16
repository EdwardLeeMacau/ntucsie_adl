import json
import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# Reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    dataloaders: Dict[str, DataLoader] = {
        TRAIN: DataLoader(
            datasets[TRAIN], batch_size=args.batch_size,
            shuffle=True, collate_fn=datasets[TRAIN].collate_fn,
            num_workers=8,
        ),
        DEV: DataLoader(
            datasets[DEV], batch_size=args.batch_size,
            collate_fn=datasets[DEV].collate_fn, num_workers=8
        )
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(intent2idx)
    )
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50.0, gamma=0.1)

    criteria = torch.nn.CrossEntropyLoss(reduction='sum')

    best = 0
    for epoch in range(args.num_epoch):
        # Training loop
        model.train()
        with tqdm(dataloaders[TRAIN], desc=f"Train: {epoch+1:3d}/{args.num_epoch:3d}") as pbar:
            train_acc = 0
            count = 0
            for data in pbar:
                x, y = data
                n = y.shape[0]

                x['token'] = x['token'].to(args.device)
                y = y.to(args.device)

                optimizer.zero_grad()

                predict = model(x)
                loss = criteria(predict, y)
                loss.backward()

                train_acc = torch.sum(torch.argmax(predict, dim=1) == y) / n

                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item() / n:.6f}", acc=f"{train_acc:.2%}")

        scheduler.step()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_acc = 0
            count = 0
            with tqdm(dataloaders[DEV], desc="Validation: ") as pbar:
                for data in pbar:
                    x, y = data
                    count += y.shape[0]

                    x['token'] = x['token'].to(args.device)
                    y = y.to(args.device)

                    predict = model(x)
                    loss = criteria(predict, y)
                    dev_loss += loss.item()
                    dev_acc += torch.sum(torch.argmax(predict, dim=1) == y)

                    pbar.set_postfix(loss=f"{dev_loss / count:.6f}", acc=f"{dev_acc / count:.2%}")

        if dev_acc > best:
            best = dev_acc
            model.cpu()
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best.pt"))
            model = model.to(args.device)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
