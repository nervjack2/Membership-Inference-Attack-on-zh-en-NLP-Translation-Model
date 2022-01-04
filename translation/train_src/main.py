import transformers
import torch  
import argparse
from hyper import hp
import tqdm as tqdm 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from data import TrainingDataset 
from utils import load_dataset, compute_metrics

def main(
    train_path: str,
    dev_path: str,
    save_dir: str,
    model_checkpoint: str
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data, dev_data = load_dataset(train_path, dev_path)
    # Define tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "en-XX"
    # If model is in family of t5, add prefix in input
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "translate English to Romanian: "
    else:
        prefix = ""
    train_dataset = TrainingDataset(train_data, prefix, tokenizer)
    dev_dataset = TrainingDataset(dev_data, prefix, tokenizer) 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
    model_args = Seq2SeqTrainingArguments(
        save_dir,
        evaluation_strategy = "steps",
        eval_steps = 500,
        learning_rate = hp.lr,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=hp.batch_size,
        weight_decay=hp.weight_decay,
        save_total_limit=3,
        num_train_epochs=hp.epoch,
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True
    )
    trainer = Seq2SeqTrainer(
        model,
        model_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_checkpoint", type=str, default="Helsinki-NLP/opus-mt-zh-en")
    args = parser.parse_args()
    main(**vars(args))
