import json
from os.path import join


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

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result