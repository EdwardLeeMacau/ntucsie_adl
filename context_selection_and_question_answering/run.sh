# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python predict.py \
    --pad_to_max_length \
    --multiple_choice_model ./ckpt/multiple_choice/bert-base-chinese.best \
    --question_answering_model ./ckpt/question_answering/hfl-chinese-roberta-wwm-ext.best \
    --per_device_eval_batch_size 1 \
    --context_file $1 \
    --question_file $2 \
    --predict_file $3
