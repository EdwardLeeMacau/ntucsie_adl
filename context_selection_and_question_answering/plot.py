import json
import os

import evaluate
# predict.question_answering() only return accuracy (EM) and f1-score
from predict import parse_args, question_answering, question_answering_loss

def main():
    args = parse_args()

    with open(args.context_file, encoding='utf-8') as f:
        context = json.load(f)

    with open(args.question_file, encoding='utf-8') as f:
        questions = json.load(f)

    metric = evaluate.load("squad")
    root_dir = args.question_answering_model
    for directory in sorted(os.listdir(root_dir)):
        if not directory.startswith('step_'):
            continue

        args.question_answering_model = os.path.join(root_dir, directory)
        print(f"Load checkpoint from {args.question_answering_model}")

        answers = question_answering(context, questions, args)
        score = metric.compute(predictions=answers.predictions, references=answers.label_ids)

        loss = question_answering_loss(context, questions, args)
        score['loss'] = loss

        print(score)

if __name__ == "__main__":
    main()