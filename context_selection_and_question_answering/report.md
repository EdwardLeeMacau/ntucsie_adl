# Report

Student ID: R11922001

## Q1: Data Preprocessing

### Tokenizer

According to the [document](https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/bert#transformers.BertTokenizer), Huggingface implements WordPiece as default tokenizer for BERT. [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) is a commonly used techniques to segment words into sub-words.

1. It first initialize the word unit inventory with the basic characters (single word in Chinese)
2. Build a language model on the training data using the inventory from 1.
3. Generate a new word unit by combining two units out of the current word inventory to increment the word unit inventory by one. Choose the new word unit out of all possible ones that increases the likelihood on the training data the most when added to the model.
4. Repeat 2 and 3 until vocab size meets stop criteria
   - Predefined limit of word units, or
   - The likelihood increase falls below a certain threshold.

### Answer Span

Position-on-characters to Position-on-token

2b. Post-processing

Reference:
- [ZhiHu - SQuAD Question Answering based on BERT pre-trained model](https://zhuanlan.zhihu.com/p/473157694?utm_id=0)

---

## Q2: Modeling with BERTs and their variants

The model can be separated as 2 parts, one for multiple choices and another for question answering.

- Multiple choice:
  - Pretrained transformer: bert-base-chinese
  - Model configuration:
    - attention_probs_dropout_prob: 0.1,
    - classifier_dropout: null,
    - directionality: bi-directional,
    - hidden_act: gelu,
    - hidden_dropout_prob: 0.1,
    - hidden_size: 768,
    - initializer_range: 0.02,
    - intermediate_size: 3072,
    - layer_norm_eps: 1e-12,
    - max_position_embeddings: 512,
    - model_type: bert,
    - num_attention_heads: 12,
    - num_hidden_layers: 12,
    - pad_token_id: 0,
    - pooler_fc_size: 768,
    - pooler_num_attention_heads: 12,
    - pooler_num_fc_layers: 3,
    - pooler_size_per_head: 128,
    - pooler_type: first_token_transform,
    - position_embedding_type: absolute,
    - torch_dtype: float32,
    - transformers_version: 4.22.2,
    - type_vocab_size: 2,
    - use_cache: true,
    - vocab_size: 21128
  - Loss function:
  - Batch size and optimizer:
    - `Adam(lr=3e-5, weight_decay=0)`
    - Max epoch: 1
    - Batch size: 2
    - Gradient accumulation steps: 4
  - Model accuracy: 95.78%

- Question Answering:
  - Pretrained transformer: hfl/chinese-roberta-wwm-ext
  - Model configuration:
    - attention_probs_dropout_prob: 0.1,
    - bos_token_id: 0,
    - classifier_dropout: null,
    - directionality: bi-directional,
    - eos_token_id: 2,
    - hidden_act: gelu,
    - hidden_dropout_prob: 0.1,
    - hidden_size: 768,
    - initializer_range: 0.02,
    - intermediate_size: 3072,
    - layer_norm_eps: 1e-12,
    - max_position_embeddings: 512,
    - model_type: bert,
    - num_attention_heads: 12,
    - num_hidden_layers: 12,
    - output_past: true,
    - pad_token_id: 0,
    - pooler_fc_size: 768,
    - pooler_num_attention_heads: 12,
    - pooler_num_fc_layers: 3,
    - pooler_size_per_head: 128,
    - pooler_type: first_token_transform,
    - position_embedding_type: absolute,
    - torch_dtype: float32,
    - transformers_version: 4.22.2,
    - type_vocab_size: 2,
    - use_cache: true,
    - vocab_size: 21128
  - Loss function:
  - Batch size and optimizer:
    - `Adam(lr=3e-5, weight_decay=0)`
    - Max epoch: 15
    - Batch size: 8
    - Gradient accumulation steps: 8
  - Model accuracy: 81.62%

Because we need to solve 2 problems sequently, the accuracy equal to the multiplication of both models.
- Evaluation accuracy: 78.18%
- Public score (2 highest submissions): 79.475% / 77.124%

1. Try another type of pretrained model and describe

---

## Q3: Learning curves of QA model

---

## Q4: Pretrained v.s. Not Pretrained

---

## Q5: Bonus: HW1 with BERTs
