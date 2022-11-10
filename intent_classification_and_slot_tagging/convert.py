import argparse
import json
import os
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Fine tune a transformers model on a text classification task.")
    parser.add_argument(
        "--data_file", type=str, default=None, help="A csv or a json file containing the question data."
    )
    parser.add_argument(
        "--index_file", type=str, default=None, help="A csv or a json file containing the mapping."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the converted dataset."
    )

    return parser.parse_args()

def to_cola_format(intent2idx: List[str], data: List[Dict]) -> List[Dict]:
    results = []

    for idx, q in enumerate(data):
        q_formatted = {
            'idx': idx,
            'sentence': q['text'],
        }

        # Handling test set, the field 'relevant' does not exist.
        if 'intent' in q:
            q_formatted['label'] = intent2idx[q['intent']]

        results.append(q_formatted)

    return results


def main():
    args = parse_args()

    with open(args.index_file, encoding='utf-8') as f:
        intent2idx = json.load(f)

    with open(args.data_file, encoding='utf-8') as f:
        data = json.load(f)

    # Create directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.data_file)

    # Wrap question in SWAG format.
    cola = to_cola_format(intent2idx, data)

    # Output file for multiple choice task
    with open(os.path.join(args.output_dir, f"{basename}"), 'w') as f:
        json.dump(cola, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
