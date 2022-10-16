import json
import pickle
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=1, collate_fn=dataset.test_collate_fn, num_workers=8,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, bidirectional=args.bidirectional, num_class=9
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)

    # predict dataset
    with open(args.pred_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tags'])
        for x in tqdm(dataloader):
            x['token'] = x['token'].to(args.device)
            y = torch.argmax(model(x), dim=2)[0].cpu().tolist()
            writer.writerow([x['id'][0], " ".join(map(lambda idx: dataset.idx2label(idx), y))])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)