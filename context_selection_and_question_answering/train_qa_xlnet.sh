# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python run_qa_beam_search_no_trainer.py \
  --model_name_or_path hfl/chinese-xlnet-base \
  --train_file ./cache/train_qa.json \
  --validation_file ./cache/valid_qa.json \
  --output_dir ./ckpt/question_answering/hfl-chinese-xlnet-base \
  --pad_to_max_length \
  --seed 0 \
  --with_tracking \
  --checkpointing_steps 200 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 250 \
  --gradient_accumulation_steps 8

