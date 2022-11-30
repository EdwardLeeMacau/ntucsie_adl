# Possible strategies:
# greedy decoding by calling greedy_search():
#   if num_beams=1 and do_sample=False.
# contrastive search by calling contrastive_search()
#   if penalty_alpha>0 and top_k>1
# multinomial sampling by calling sample()
#   if num_beams=1 and do_sample=True.
# beam-search decoding by calling beam_search()
#   if num_beams>1 and do_sample=False.
# beam-search multinomial sampling by calling beam_sample()
#   if num_beams>1 and do_sample=True.
# diverse beam-search decoding by calling group_beam_search()
#   if num_beams>1 and num_beam_groups>1.
#
# (Implement related functions)
# constrained beam-search decoding by calling constrained_beam_search()
#   if constraints!=None or force_words_ids!=None.

# Greedy Search
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/greedy \
    --num_beams 1 \
    --seed 0

# Multinomial
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/multinomial \
    --do_sample \
    --seed 0

# Beam-Search(beam=3)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-3 \
    --num_beams 3 \
    --seed 0

# Beam-Search(beam=5)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-5 \
    --num_beams 5 \
    --seed 0

# Beam-Search(beam=7)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-7 \
    --num_beams 7 \
    --seed 0

# Beam-Search-MultiNomial(beam=3)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-multinomial-3 \
    --num_beams 3 \
    --do_sample \
    --seed 0

# Beam-Search-MultiNomial(beam=5)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-multinomial-5 \
    --num_beams 5 \
    --do_sample \
    --seed 0

# Beam-Search-MultiNomial(beam=7)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/beam-search-multinomial-7 \
    --num_beams 7 \
    --do_sample \
    --seed 0

# Top-p (p=0.90)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-p-0.90 \
    --top_p 0.90 \
    --do_sample \
    --seed 0

# Top-p (p=0.92)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-p-0.92 \
    --top_p 0.92 \
    --do_sample \
    --seed 0

# Top-p (p=0.95)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-p-0.95 \
    --top_p 0.95 \
    --do_sample \
    --seed 0

# Top-k (k=3)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-k-3 \
    --top_k 3 \
    --do_sample \
    --seed 0

# Top-k (k=5)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-k-5 \
    --top_k 5 \
    --do_sample \
    --seed 0

# Top-k (k=10)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-k-10 \
    --top_k 10 \
    --do_sample \
    --seed 0

# Top-k (k=20)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-k-20 \
    --top_k 20 \
    --do_sample \
    --seed 0

# Top-k (k=50)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/top-k-50 \
    --top_k 50 \
    --do_sample \
    --seed 0

# Temperature(T=0.6)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/temperature-0.6 \
    --top_k 0 \
    --temperature 0.6 \
    --do_sample \
    --seed 0

# Temperature(T=0.7)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/temperature-0.7 \
    --top_k 0 \
    --temperature 0.7 \
    --do_sample \
    --seed 0

# Temperature(T=0.9)
CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict ./ckpt/google-mt5-small/predictions/temperature-0.9 \
    --top_k 0 \
    --temperature 0.9 \
    --do_sample \
    --seed 0
