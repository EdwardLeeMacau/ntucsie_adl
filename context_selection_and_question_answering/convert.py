import argparse
import json
import math
import os
import random
import copy
from typing import Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fine tune a transformers model on a multiple choice task")
    parser.add_argument(
        "--question_file", type=str, default=None, help="A csv or a json file containing the question data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None, help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.0,
        help="Ratio of augmented data. Default disabled 0.0"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the converted dataset."
    )

    return parser.parse_args()

def to_swag_format(context: List[str], questions: List[Dict]) -> List[Dict]:
    results = []

    for q in questions:
        q_formatted = {
            'video-id': q['id'],
            'fold-ind': q['id'],
            'sent1': q['question'],
            'sent2': '',
            'title': '',
            'gold-source': "gold",
            'ending0': context[q['paragraphs'][0]],
            'ending1': context[q['paragraphs'][1]],
            'ending2': context[q['paragraphs'][2]],
            'ending3': context[q['paragraphs'][3]],
        }

        # Handling test set, the field 'relevant' does not exist.
        if 'relevant' in q:
            q_formatted['label'] = q['paragraphs'].index(q['relevant'])

        results.append(q_formatted)

    return results

def to_squad_format(context: List[str], questions: List[Dict]) -> List[Dict]:
    results = []
    for q in questions:
        q_formatted = {
            'id': q['id'],
            'question': q['question'],
            'title': '',
        }

        q_formatted['context'] = context[q['relevant']]

        # Handling test set, the field 'answers' does not exist.
        if 'answer' in q:
            q_formatted['answers'] = {
                'answer_start': [q['answer']['start']],
                'text': [q['answer']['text']]
            }

        results.append(q_formatted)

    return results

def augment(context: List[str], questions: List[Dict], ratio: float = 0.0) -> List[Dict]:
    size = math.ceil(len(questions) * ratio)

    # Re-sample question
    questions: List[Dict] = [copy.deepcopy(q) for q in np.random.choice(questions, size)]
    possible_choices = np.arange(len(context))

    for question in questions:
        # Re-sample relevant
        samples = np.random.choice(
            possible_choices[possible_choices != question['relevant']], size=3, replace=False
        ).tolist()

        assert (question['relevant'] not in samples)

        samples = samples + [question['relevant']]
        random.shuffle(samples)

        question['paragraphs'] = samples

    return questions

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with open(args.context_file, encoding='utf-8') as f:
        context = json.load(f)

    with open(args.question_file, encoding='utf-8') as f:
        questions = json.load(f)

    # Create directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.question_file)
    stem, ext = os.path.splitext(basename)

    # Wrap question in SWAG format.
    swag: List = to_swag_format(context, questions)
    if args.ratio != 0.0:
        swag.extend(to_swag_format(context, augment(context, questions, args.ratio)))

    # Output file for multiple choice task
    with open(os.path.join(args.output_dir, f"{stem}_mc{ext}"), 'w') as f:
        json.dump(swag, f, ensure_ascii=False, indent=4)

    # Wrap question in SQuAD format.
    squad: List = to_squad_format(context, questions)

    # Output file for question answering task
    with open(os.path.join(args.output_dir, f"{stem}_qa{ext}"), 'w') as f:
        json.dump(squad, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
