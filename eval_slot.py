import json
import pickle
from argparse import ArgumentParser, Namespace
from itertools import starmap
from pathlib import Path
from typing import Dict

import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

DEV = "eval"
SPLITS = [DEV, ]

@torch.no_grad()
def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    path = args.data_dir / f"eval.json"
    data = json.loads(path.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(tag2idx)
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    x, y = next(iter(dataloader))
    mask, lengths = x['mask'], x['length']

    predict = torch.argmax(model(x), dim=2)
    correct = (predict == y)
    correct[~mask] = True
    joint_acc = torch.sum(torch.all(correct, dim=1)) / y.shape[0]

    predict, label = predict.tolist(), y.tolist()
    predict = list(starmap(
        lambda l, pred: [dataset.idx2label(i) for i in pred[:l]],
        zip(lengths, predict)
    ))
    label = list(starmap(
        lambda l, pred: [dataset.idx2label(i) for i in pred[:l]],
        zip(lengths, label)
    ))

    print(f"Joint accuracy: {joint_acc:.2%}")
    print("Classification report:")
    print(classification_report(
        predict, label, scheme=IOB2, mode='strict'
    ))

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
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
