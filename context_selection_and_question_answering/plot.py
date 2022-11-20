import json
import os

import numpy as np
from matplotlib import pyplot as plt

def main():
    roberta = {}
    with open('./data/hfl-chinese-roberta-wwm-ext_qa_no_trainer_train_loss.json') as f:
        roberta['train_loss'] = np.array(json.load(f))[:, 1:]

    with open('./data/hfl-chinese-roberta-wwm-ext_qa_no_trainer_val_loss.json') as f:
        roberta['val_loss'] = np.array([(m['step'], m['loss'] / 3941) for m in json.load(f)])

    with open('./data/hfl-chinese-roberta-wwm-ext_qa_no_trainer_exact_match.json') as f:
        roberta['exact_match'] = np.array(json.load(f))[:, 1:]

    scratch = {}
    with open('./data/bert_base_train_loss.json') as f:
        scratch['train_loss'] = np.array(json.load(f))[:, 1:]

    with open('./data/bert_base_exact_match.json') as f:
        scratch['exact_match'] = np.array(json.load(f))[:, 1:]

    plt.figure(dpi=300, figsize=(12.8, 7.2))
    plt.xlabel('Steps')

    x, y = roberta['train_loss'][:, 0], roberta['train_loss'][:, 1]
    line1 = plt.plot(x, y, label='Train Loss', linewidth=0.25, linestyle='--')

    x, y = roberta['val_loss'][:, 0], roberta['val_loss'][:, 1]
    line2 = plt.plot(x, y, label="Validation Loss", color='g')
    plt.ylabel('Loss')

    plt.twinx()

    x, y = roberta['exact_match'][:, 0], roberta['exact_match'][:, 1]
    plt.ylim((-5, 100))
    plt.ylabel('Exact Match')
    line3 = plt.plot(x, y, 'g--', label='Validation Accuracy')

    plt.legend(line1 + line2 + line3, ['Train loss', 'Validation loss', 'Exact match'])
    plt.title('Learning Curve')
    plt.savefig('curve.jpeg')
    plt.clf()

    # Q4: Train from scratch

    plt.figure(dpi=300, figsize=(12.8, 7.2))
    plt.xlabel('Steps')

    x, y = scratch['exact_match'][:, 0], scratch['exact_match'][:, 1]
    line1 = plt.plot(x, y, label='From scratch')

    x, y = roberta['exact_match'][:, 0], roberta['exact_match'][:, 1]
    line2 = plt.plot(x, y, label='chinese-roberta')
    plt.ylim((-5, 100))
    plt.ylabel('Exact Match')

    plt.legend(line1 + line2, ['From scratch', 'chinese-roberta'])
    plt.title('Performance comparison between pretrained model and training from scratch')
    plt.savefig('q4.jpeg')
    plt.clf()

if __name__ == "__main__":
    main()