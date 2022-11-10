python convert.py \
    --context_file data/context.json \
    --question_file data/train.json \
    --seed 0 \
    --output_dir cache

python convert.py \
    --context_file data/context.json \
    --question_file data/valid.json \
    --output_dir cache

python convert.py \
    --context_file data/context.json \
    --question_file data/train_val.json \
    --seed 0 \
    --output_dir cache
