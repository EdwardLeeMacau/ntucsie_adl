import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from utils import Vocab, pad_to_len

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Tuple[Dict, torch.Tensor]:
        # Split string by SPACE
        x = [d['text'].split(' ') for d in samples]
        y = torch.tensor([self.label2idx(d['intent']) for d in samples], dtype=torch.long)
        token = torch.tensor(self.vocab.encode_batch(x, self.max_len), dtype=torch.long)

        x_length = torch.tensor([len(sentence) for sentence in x], dtype=torch.long).flatten()
        x_id = [d['id'] for d in samples]

        return ({ 'token': token, 'length': x_length, 'id': x_id }, y)

    def test_collate_fn(self, samples: List[Dict]) -> Dict:
        # Split string by SPACE
        x = [d['text'].split(' ') for d in samples]
        token = torch.tensor(self.vocab.encode_batch(x, self.max_len), dtype=torch.long)

        x_length = torch.tensor([len(sentence) for sentence in x], dtype=torch.long).flatten()
        x_id = [d['id'] for d in samples]

        return { 'token': token, 'length': x_length, 'id': x_id }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples: List[Dict]) -> Tuple[Dict, torch.Tensor]:
        # Split string by SPACE
        x = [s['tokens'] for s in samples]
        y = [[self.label2idx(tag) for tag in d['tags']] for d in samples]

        to_len = max(len(tag) for tag in y) if self.max_len is None else self.max_len
        length = torch.tensor([len(sentence) for sentence in x], dtype=torch.long).flatten()

        token = torch.tensor(self.vocab.encode_batch(x, self.max_len), dtype=torch.long)
        tags = torch.tensor(pad_to_len(y, to_len, 0), dtype=torch.long)[:, :max(length)]

        mask = (token != self.vocab.pad_id)[:, :max(length)]
        ids = [d['id'] for d in samples]

        return ({ 'token': token, 'length': length, 'mask': mask, 'id': ids }, tags)

    def test_collate_fn(self, samples: List[Dict]) -> Dict:
        # Split string by SPACE
        x = [d['tokens'] for d in samples]
        token = torch.tensor(self.vocab.encode_batch(x, self.max_len), dtype=torch.long)

        length = torch.tensor([len(sentence) for sentence in x], dtype=torch.long).flatten()
        mask = torch.tensor([len(sentence) for sentence in x], dtype=torch.long).flatten()
        ids = [d['id'] for d in samples]

        return { 'token': token, 'length': length, 'mask': mask, 'id': ids }

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

    args = parser.parse_args()
    return args


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, None)
        for split, split_data in data.items()
    }
    dataloaders: Dict[str, DataLoader] = {
        TRAIN: DataLoader(
            datasets[TRAIN], batch_size=len(datasets[TRAIN]),
            shuffle=False, collate_fn=datasets[TRAIN].collate_fn,
        ),
        DEV: DataLoader(
            datasets[DEV], batch_size=len(datasets[DEV]),
            collate_fn=datasets[DEV].collate_fn,
        )
    }

    x, y = next(iter(dataloaders[TRAIN]))
    length = x['length'].float()
    cls_count = torch.bincount(y).float()
    # len_count = torch.bincount(length).float()
    print("Train: ")
    print(f"class mean and std: {torch.mean(cls_count)}, {torch.std(cls_count)}")
    print(f"length mean and std: {torch.mean(length)}, {torch.std(length)}")

    x, y = next(iter(dataloaders[DEV]))
    length = x['length'].float()
    cls_count = torch.bincount(y).float()
    # len_count = torch.bincount(length).float()
    print("Validation: ")
    print(f"class mean and std: {torch.mean(cls_count)}, {torch.std(cls_count)}")
    print(f"length mean and std: {torch.mean(length)}, {torch.std(length)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
