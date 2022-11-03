# ADL Fall 2022 - Homework 1

## Environment

```bash
conda env update --file environment.yml
conda activate adl-hw1
pip install -r requirements.in
```

## Download Pre-Trained Model

```bash
# To download pre-trained models for intent detection and slot tagging
# The model will be stored in ./ckpt and ./cache
bash download.sh
```

## Model Inference

```bash
# Intent Classification
python test_intent.py --ckpt_path ./ckpt/intent/best.pt
```

```bash
# Slot Tagging
python test_slot.py --ckpt_path ./ckpt/slot/best.pt
```

## Model Training

### Preprocessing

```bash
# To preprocess intent classification and slot tagging datasets
bash preprocess.sh
```

### Run Training Script

Notes: Check supported arguments by adding `--help` after the command.

```bash
# Intent Classification
python train_intent.py
```

```bash
# Slot Tagging
python train_slot.py
```
