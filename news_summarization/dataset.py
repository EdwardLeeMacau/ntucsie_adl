import argparse
import json
import jsonlines
from torch.utils.data import Dataset
from typing import Dict, List

class NewsSummarizationDataset(Dataset):
    data: List
    split: str

    def __init__(self, fname: str, split: str = "train"):
        super().__init__()

        self.split = split
        with jsonlines.open(fname, 'r') as reader:
            self.data = [obj for obj in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="File format transformation.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to jsonlines file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to json file')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = NewsSummarizationDataset(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset.data, f, indent=4)

if __name__ == "__main__":
    main()