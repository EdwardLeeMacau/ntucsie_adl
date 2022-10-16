# Report

Student ID: R11922001

## Preprocessing

I use the preprocess script provided in sample code. The details of preprocessing scripts are mentioned below:
- It collects all output labeled in training set and validation set, assign them with unique ID and store to JSON file.
  - See the key-value pairs in `intent2idx.json` and `tag2idx.json`.
- Then, it counts the frequency of words appeared in input sentences, picks the 10000 most frequently used words, assigning them unique ID also and saved as key-value pairs (e.g. "run" -> 147).
  - See `Vocab.token2idx`.
- Then, querying the word embedding of frequenly used words in Glove and store key-value pairs into `embeddings.pt` (e.g. 147 -> [0.3000 0.2544 0.7684]). Assign random vector for the words not included in pretrained word embedding.
  - See `torch.nn.Embedding` and `SeqClassifier.embed`.
- Finally, we use generated files `vocab.pkl` and `embeddings.pt` to train the model. When a sentence is fed into the model, it's first split to words by space, mapping to tokens by table look-up (`Vocab.pkl`), then mapping to tensor by table look-up (`embeddings.pt`) again.
- Notes:
  - For the words not seen in test phase, a predefined token `UNK` is assigned.
  - To pad the sentences, a predefined token `PAD` is assigned.
  - Both `PAD` and `UNK` have an embedding! They are defined in preprocessing phase.

Statistics (Token covered):
- Applied word embedding: `glove.840B.300d.txt`
- (Max) Vocab size: 10000
  - Intent Classification: 5435 / 6491 = 83.73%
  - Slot Taggin: 3000 / 4147 = 72.34%

---

## Intent Classification Model

- Model Description:
  - Feature extractor: ElmanRNN
    - $o_t, h_t = \mathrm{RNN}(w_t, h_{t-1})$, where $w_t$ is the word embedding of t-th token.
    - Keep the output for last token ($\tanh{o_T}$) and feed to classifier.
  - Classifier: Single Fully Connected Layer
    - $y = o_TA^T + b$
    - Return $\mathrm{idx} = \argmax_i y_i$ as the result of classification.
- Model Training Information:
  - Model size:
    - Hidden size: 512
    - Number of RNN layers: 3
    - Bidirectional: True
    - FC Layers: 1
  - Public score: 0.90088
  - Loss function: CrossEntropy
  - Optimizer:
    - Adam(lr=1e-3, weight_decay=1e-4)
    - Max epoch: 100
    - Keep model parameter with the highest validation accuracy
  - Batch size: 2048
  - Dropout: 0.1

---

## Slot Tagging Model

- Model Description:
  - Feature extractor: ElmanRNN
    - $o_t, h_t = \mathrm{RNN}(w_t, h_{t-1})$, where $w_t$ is the word embedding of t-th token.
    - Keep the output of each tokens ($O = \{o_1, o_2, ...o_T\}$) and feed to classifier.
  - Classifier: Single Fully Connected Layer
    - $y_t = o_tA^T + b$
    - Return $\mathrm{idx} = \argmax_i y_i$ as the result of classification.
- Model Training Information:
  - Model size:
    - Hidden size: 512
    - Number of RNN layers: 3
    - Bidirectional: True
    - FC Layers: 1
  - Public score: 0.79356
  - Loss function: FocalLoss(alpha=0.25, gamma=10)
  - Optimizer:
    - Adam(lr=1e-3, weight_decay=1e-4)
    - Max epoch: 100
    - Keep model parameter with the highest validation accuracy
  - Batch size: 512
  - Dropout: 0.1

---

## Sequence Tagging Evaluation

```
Classification report:
              precision    recall  f1-score   support

        date       0.76      0.75      0.76       209
  first_name       0.89      0.94      0.91        97
   last_name       0.78      0.90      0.84        68
      people       0.71      0.71      0.71       241
        time       0.78      0.82      0.80       209

   micro avg       0.77      0.79      0.78       824
   macro avg       0.79      0.82      0.80       824
weighted avg       0.77      0.79      0.78       824
```

For example, the report shows that my pre-trained model has precision = 0.78, recall = 0.90, f1-score = 0.84 and support = 68 on class "last_name", means that:
- 78% tokens are correctly classified as "last_name" in test cases with class "last_name".
- 90% tokens in class "last_name" are detected from all test cases.
- Considering precision and recall in both, the model has a performance score (f1-score is selected here) 0.84 in the class "last_name", where the score is the harmonic mean of precision and recall.

There are multiple ways to average the performance on difference classes, 'mirco' (precision and recall are weighted by number of cases), 'macro' (precision and recall for each classes are NOT weighted) and 'weighted' (self-defined weight for difference classes, not specified here).

TODO: Token accuracy

TODO: Joint accuracy

Notes:
- $F_\beta$-score provides us a single goal (index) to represent model performance. When $\beta = 1$, it takes average precision and recall with equal weight.

Reference:
- [seqeval document](https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py#L41).


---

## Compare With Different Configuration