#!/bin/bash

# Download pre-trained model for intent classification
if [ ! -d ./ckpt/intent ]; then
  mkdir -p ./ckpt/intent
fi

if [ ! -f ./ckpt/intent/best.pt ]; then
  wget https://www.dropbox.com/s/om3it8mo75yfu1k/best.intent.pt?dl=1 -O ./ckpt/intent/best.pt
fi

if [ ! -d ./cache/intent ]; then
  mkdir -p ./cache/intent
fi

if [ ! -f ./cache/intent/embeddings.pt ]; then
  wget https://www.dropbox.com/s/jy1vhetisvr4n5x/embeddings.pt?dl=1 -O ./cache/intent/embeddings.pt
fi

if [ ! -f ./cache/intent/intent2idx.json ]; then
  wget https://www.dropbox.com/s/u87c3nldpqikf0i/intent2idx.json?dl=1 -O ./cache/intent/intent2idx.json
fi

if [ ! -f ./cache/intent/vocab.pkl ]; then
  wget https://www.dropbox.com/s/ggn7ydbpocjqw0l/vocab.pkl?dl=1 -O ./cache/intent/vocab.pkl
fi

# Download pre-trained model for slot tagging
if [ ! -d ./ckpt/slot ]; then
  mkdir -p ./ckpt/slot
fi

if [ ! -f ./ckpt/slot/best.pt ]; then
  wget https://www.dropbox.com/s/k3wipx8ctjtcj2u/best.slot.pt?dl=1 -O ./ckpt/slot/best.pt
fi

if [ ! -d ./cache/slot ]; then
  mkdir -p ./cache/slot
fi

if [ ! -f ./cache/slot/embeddings.pt ]; then
  wget https://www.dropbox.com/s/y8m3hhsao8tx24s/embeddings.pt?dl=1 -O ./cache/slot/embeddings.pt
fi

if [ ! -f ./cache/slot/tag2idx.json ]; then
  wget https://www.dropbox.com/s/lmb6rs7ynfh8st7/tag2idx.json?dl=1 -O ./cache/slot/tag2idx.json
fi

if [ ! -f ./cache/slot/vocab.pkl ]; then
  wget https://www.dropbox.com/s/hj47y85drojfaw7/vocab.pkl?dl=1 -O ./cache/slot/vocab.pkl
fi
