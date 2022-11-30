# ADL Fall 2022 - Homework 3

## Environment

```bash
conda env update --file environment.yml
conda activate adl-hw3
pip install -r requirements.in
```

## Download Pre-Trained Model

```bash
# To download pre-trained models for news summarization
# The model will be stored in ./ckpt
bash download.sh
```

## Preprocessing

### Convert JSON lines to JSON

```bash
$ python dataset.py -i <jsonl-fname> -o <json-fname>
```

## Model Inference

```bash
# ${1}: Path to the input file
# ${2}: Path to the output file
$ ./run.sh $1 $2
```


## Model Training

```bash
# ${1}: Path to the training data (converted to .json format)
# ${2}: Path to the validation data (converted to .json format)
$ ./train.sh $1 $2
```

