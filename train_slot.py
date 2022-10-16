import json
import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import FocalLoss, SeqTagger
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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
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
            collate_fn=datasets[DEV].collate_fn, num_workers=8,
        )
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(tag2idx)
    )
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    criteria = FocalLoss(gamma=10, reduction='sum')

    best = 0
    for epoch in range(args.num_epoch):
        # Training loop
        model.train()
        with tqdm(dataloaders[TRAIN], desc=f"Train: {epoch+1:3d}/{args.num_epoch:3d}") as pbar:
            for data in pbar:
                x, y = data
                mask = x['mask']
                n = y.shape[0]
                n_tokens = torch.sum(mask)

                # mask = mask.to(args.device)
                x['token'] = x['token'].to(args.device)
                y = y.to(args.device)

                optimizer.zero_grad()

                predict = model(x)
                loss = criteria(predict[mask], F.one_hot(y, num_classes=len(tag2idx)).float()[mask])
                loss.backward()

                correct = (torch.argmax(predict, dim=2) == y)
                correct[~mask] = True
                joint_acc = torch.sum(torch.all(correct, dim=1)) / n

                correct = correct[mask]
                token_acc = torch.sum(correct) / n_tokens
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item() / n:.6f}", token_acc=f"{token_acc:.2%}", joint_acc=f"{joint_acc:.2%}")

        scheduler.step()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_acc = 0
            batches_count = 0
            tokens_count = 0
            with tqdm(dataloaders[DEV], desc="Validation: ") as pbar:
                for data in pbar:
                    x, y = data
                    mask = x['mask']

                    batches_count += y.shape[0]
                    tokens_count += torch.sum(mask)

                    x['token'] = x['token'].to(args.device)
                    y = y.to(args.device)

                    predict = model(x)
                    loss = criteria(predict[mask], F.one_hot(y, num_classes=len(tag2idx)).float()[mask])
                    dev_loss += loss.item()

                    correct = (torch.argmax(predict, dim=2) == y)
                    correct[~mask] = True
                    dev_acc += torch.sum(torch.all(correct, dim=1))

                    pbar.set_postfix(loss=f"{dev_loss / tokens_count:.6f}", joint_acc=f"{dev_acc / batches_count:.2%}")

        if dev_acc > best:
            best = dev_acc
            model.cpu()
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"best.pt"))
            model = model.to(args.device)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
