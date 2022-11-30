# Report

Student ID: R11922001

## Q1: Model

### Model Parameters

T5 model is selected for text generating. Shortly say, it's an encoder-decoder model, which decoder is trained to output the probability distribution of next word given previous words as output.

Detail parameters is listed below.

```
T5Config {
  "_name_or_path": "google/mt5-small",
  "architectures": [
    "MT5ForConditionalGeneration"
  ],
  "d_ff": 1024,
  "d_kv": 64,
  "d_model": 512,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 8,
  "num_heads": 6,
  "num_layers": 8,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "tokenizer_class": "T5Tokenizer",
  "transformers_version": "4.22.2",
  "use_cache": true,
  "vocab_size": 250112
}
```

<div style='page-break-after:always'></div>

### Preprocessing (Tokenization / Data Cleaning and others)

According to the [document](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer), Huggingface implements [SentencesPiece](https://github.com/google/sentencepiece), which based on subword units, byte-pair-encoding and unigram language model, as default tokenizer for T5.

Except the default tokenizer, no more data preprocessing techniques, such as data cleaning, are applied.

---

## Q2: Training

### Hyperparameters

- Pretrained model: `google/mt5-small` from HuggingFace
- Batch size: 12 (Gradient accumulation x batch size)
- Epochs: Total 30 epochs, stops at 43500 steps.
- Optimizer:
  - Type: AdamW
  - Learning rate: 2e-4
  - Learning rate scheduling: linear
  - Weight Decay: 0
- Max source length: 256
- Max target length: 64
- Source Prefix: "summarize: "

---

<div style='page-break-after:always'></div>

### Learning Curve

**Rouge scores**

![](./ckpt/google-mt5-small/predictions/metrics.png)

**Train loss**

![](./ckpt/google-mt5-small/predictions/train-loss.png)

---

<div style='page-break-after:always'></div>

## Q3: Generation Strategies

Notes that the text generation model applied in this task is based on Auto-regressive, it aims to predict the probability of next word by the previous words.

### Strategies

- Greedy
  1. Select word that have highest probability for the current state.
  2. Feed the current word into decoder.
  3. Repeatedly until EOS
- Beam Search
  1. Compare with greedy strategy, beam search keeps track $k$-highest options in each state.
  2. It aims to maximize the combination probability $P(w_i:w_{i+k-1}) = P(w_i) \times P(w_{i+1}) \times \dots \times P(w_{i+k-1})$
- Top-k and top-p sampling
  1. Compare with deterministic strategies, top-k and top-p sampling involves randomness, they both sample words from probability distribution.
  2. Top-k strategy sample words from $k$-highest bins. Instead sampling with fixed number of bins, top-p strategy sample words from bins with accumulated probability $p$.
- Temperature
  1. Before calculating probability of word, the value $z_i$ divided by temperature $T$.
  2. Division leads the distribution become sharper when $0 < T < 1$, become smoother when $1 < T$.

Reference:
- [HuggingFace - How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
- [HuggingFace - API documents](https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin)

<div style='page-break-after:always'></div>

### Hyperparameters

I've picked beam search with `num_beams=2`.

|           Strategies           | Rouge-1 (f1x100) | Rouge-2 (f1x100) | Rouge-L (f1x100) |
| :----------------------------: | :--------------: | :--------------: | :--------------: |
|             Greedy             |     25.9445      |      9.6826      |     23.2627      |
|          Multinomial           |     20.1501      |      6.0872      |     17.7440      |
| Beam Search (Multinomial, k=3) |     26.7586      |     10.4504      |     23.9635      |
| Beam Search (Multinomial, k=5) |     26.8503      |     10.6336      |     24.0998      |
| Beam Search (Multinomial, k=7) |     27.0025      |     10.8326      |     24.2474      |
|       Beam Search (k=3)        |     27.2338      |     10.9092      |     24.4151      |
|       Beam Search (k=5)        |     27.2822      |     11.1156      |     24.5003      |
|       Beam Search (k=7)        |     27.2738      |     11.1693      |     24.4847      |
|          Top-k (k=3)           |                  |                  |                  |
|          Top-k (k=5)           |                  |                  |                  |
<!-- |          Top-k (k=10)          |     19.9568      |      6.0399      |     17.6947      | -->
|         Top-p (p=0.90)         |     21.6815      |      7.0183      |     19.1558      |
|         Top-p (p=0.92)         |     18.3322      |      5.3111      |     16.2575      |
<!-- |         Top-p (p=0.95)         |     18.3083      |      5.3088      |     16.2579      | -->

