import argparse
import csv
import json
import os
from itertools import chain
from typing import Dict, List

import evaluate
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from convert import to_squad_format, to_swag_format
from datasets import Dataset
from run_swag_no_trainer import DataCollatorForMultipleChoice
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoModelForQuestionAnswering, AutoTokenizer,
                          DataCollatorWithPadding, EvalPrediction,
                          PreTrainedTokenizerBase, default_data_collator)
from transformers.utils import PaddingStrategy
from utils_qa import postprocess_qa_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiple choice and question answering script.")
    parser.add_argument(
        "--context_file", type=str, required=True,
        help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--question_file", type=str, required=True,
        help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--multiple_choice_model",
        type=str,
        help="Path to multiple choice pre-trained model",
    )
    parser.add_argument(
        "--question_answering_model",
        type=str,
        help="Path to question answering pre-trained model"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--predict_file", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Run with CPU only mode."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )

    return parser.parse_args()

def multiple_choice(context: List[str], questions: List[Dict], args: argparse.Namespace) -> List[int]:
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator(cpu=args.cpu)

    # Prepare dataset
    # When using your own dataset or a difference dataset from swag, you will probably need to change this.
    raw_datasets = Dataset.from_dict(
        pd.DataFrame(to_swag_format(context, questions)).to_dict(orient='list')
    )
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    # Load pre-trained model and tokenizer
    config = AutoConfig.from_pretrained(args.multiple_choice_model)
    tokenizer = AutoTokenizer.from_pretrained(args.multiple_choice_model)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.multiple_choice_model, from_tf=False, config = config
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets
    # First we tokenize all the texts
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )

        tokenized_inputs = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_inputs

    with accelerator.main_process_first():
        dataset = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets.column_names
        )

    # DataLoader creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    # Choice prediction
    result = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(**batch).logits.argmax(dim=-1).detach().tolist()
            result.extend(outputs)

    return result

# Returns predictions
# Can evaluate model accuracy (EM) and f1-score
def question_answering(context: List[str], questions: List[Dict], args: argparse.Namespace) -> List[str]:
    accelerator = Accelerator()

    # Prepare dataset
    # When using your own dataset or a difference dataset from swag, you will probably need to change this.
    raw_datasets = Dataset.from_dict(
        pd.DataFrame(to_squad_format(context, questions)).to_dict(orient='list')
    )

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.question_answering_model:
        config = AutoConfig.from_pretrained(args.question_answering_model)
    else:
        raise ValueError

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.question_answering_model:
        tokenizer = AutoTokenizer.from_pretrained(args.question_answering_model)
    else:
        raise ValueError

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.question_answering_model,
        from_tf=False,
        config=config
    )

    column_names = raw_datasets.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. THis key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substring of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples['example_id'] = []

        for i in range(len(tokenized_examples['input_ids'])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question)
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing the span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples['id'][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples['offset_mapping'][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    with accelerator.main_process_first():
        dataset = raw_datasets.map(
            prepare_validation_features,
            batched=True,
            remove_columns=raw_datasets.column_names
        )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataloader = DataLoader(
        dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )

        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{'id': ex['id'], 'answers': ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    all_start_logits = []
    all_end_logits = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])

    # Concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, dataset, max_len)

    outputs = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(raw_datasets, dataset, outputs)

    return prediction

# Returns model loss
def question_answering_loss(context: List[str], questions: List[Dict], args: argparse.Namespace) -> List[str]:
    accelerator = Accelerator()

    # Prepare dataset
    # When using your own dataset or a difference dataset from swag, you will probably need to change this.
    raw_datasets = Dataset.from_dict(
        pd.DataFrame(to_squad_format(context, questions)).to_dict(orient='list')
    )

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.question_answering_model:
        config = AutoConfig.from_pretrained(args.question_answering_model)
    else:
        raise ValueError

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.question_answering_model:
        tokenizer = AutoTokenizer.from_pretrained(args.question_answering_model)
    else:
        raise ValueError

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.question_answering_model,
        from_tf=False,
        config=config
    )

    column_names = raw_datasets.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    with accelerator.main_process_first():
        dataset = raw_datasets.map(
            prepare_train_features,
            batched=True,
            remove_columns=raw_datasets.column_names
        )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    dataloader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            outputs = model(**batch)
            total_loss += outputs.loss.detach().float()

    return total_loss

def main():
    args = parse_args()

    with open(args.context_file, encoding='utf-8') as f:
        context = json.load(f)

    with open(args.question_file, encoding='utf-8') as f:
        questions = json.load(f)

    if args.debug:
        args.cpu = True
        questions = questions[:10]

    choices = multiple_choice(context, questions, args)
    for question, choice in zip(questions, choices):
        question['relevant'] = question['paragraphs'][choice]

    answers = question_answering(context, questions, args).predictions
    with open(args.predict_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])

        for question, answer in zip(questions, answers):
            assert(question['id'] == answer['id'])
            writer.writerow([question['id'], answer['prediction_text']])

if __name__ == "__main__":
    main()