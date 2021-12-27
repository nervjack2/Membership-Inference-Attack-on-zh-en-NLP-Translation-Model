import json
import numpy as np 
from os.path import join
from datasets import load_metric

def load_dataset(train_path, dev_path):
    train_dict = json.load(open(train_path, 'r'))
    dev_dict = json.load(open(dev_path, 'r'))
    train_data = list(train_dict.values())
    dev_data = list(dev_dict.values())
    print(train_data[0])
    return train_data, dev_data

def load_dataset_test(data_path):
    test_dict = json.load(open(data_path, 'r'))
    test_data = list(test_dict.values())
    return test_data

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(preds, labels, tokenizer):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = load_metric("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    return round(result['bleu'], 4)