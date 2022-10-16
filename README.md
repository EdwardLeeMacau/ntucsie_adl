# ADL Fall 2022 - Homework 1

## Environment

```bash
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing

```bash
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Download Pre-Trained Model

```bash
# To download pre-trained models for intent detection and slot tagging
# The model will be stored in ./ckpt
bash download.sh
```

## Model Training

```bash
# Intent Detection
python train_intent.py
```

```bash
# Slot Tagging
python train_slot.py
```

## Model Inference

```bash
python test_intent.py --ckpt_path ./ckpt/intent/best.pt
```

```bash
# Slot Tagging
python test_slot.py --ckpt_path ./ckpt/slot/best.pt
```
