import json
import argparse
from twrouge import get_rouge


def main(args):
    refs, predictions = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            predictions[line['id']] = line['title'].strip() + '\n'

    keys = refs.keys()
    refs = [refs[key] for key in keys]
    predictions = [predictions[key] for key in keys]

    print(json.dumps(get_rouge(predictions, refs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', required=True)
    parser.add_argument('-s', '--submission', required=True)
    args = parser.parse_args()
    main(args)
