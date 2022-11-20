# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python validate.py \
    --pad_to_max_length \
    --question_answering_model ./ckpt/question_answering/hfl-chinese-roberta-wwm-ext \
    --config_name ./ckpt/question_answering/hfl-chinese-roberta-wwm-ext/config.json \
    --tokenizer_name ./ckpt/question_answering/hfl-chinese-roberta-wwm-ext \
    --per_device_eval_batch_size 1 \
    --context_file ./data/context.json \
    --question_file ./data/valid.json \
    --output_dir ./ckpt
