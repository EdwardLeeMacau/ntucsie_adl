# ADL Fall 2022 - Homework 2

## Environment

```bash
$ conda env update --file environment.yml
$ conda activate py39
$ pip install -r requirements.in
```

## Download Pre-Trained Model

```bash
# To download pre-trained models for question answering
# The model will be stored in ./ckpt
$ bash download.sh
```

## Prediction

```bash
$ bash run.sh
```

Example:

```bash
$ bash run.sh data/context.json data/test.json prediction.csv
```

## Model Training

### Preprocessing

```bash
# To convert the training data into SWAG format and SQuAD format
$ bash convert.sh
```

### Run Training Script

Notes: Check supported arguments by adding `--help` after the command.

```bash
# Multiple Choice
$ bash train_mc.sh

# Question Answering
$ bash train_qa.sh
```
