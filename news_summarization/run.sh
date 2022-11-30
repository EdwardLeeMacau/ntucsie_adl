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

# "${1}": path to the input file.
# "${2}": path to the output file.

outdir="$(dirname "$2")"

CUDA_VISIBLE_DEVICES=0 python run_summarization_no_trainer.py \
    --model_name_or_path ./ckpt/google-mt5-small \
    --num_train_epochs 0 \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --train_file $1 \
    --validation_file $1 \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --predict $outdir \
    --num_beams 5 \
    --seed 0

mv $outdir/submission.jsonl $2