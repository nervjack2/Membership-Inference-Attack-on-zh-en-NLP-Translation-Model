import transformers
import torch  
import argparse
import json 
import datasets
from tqdm import tqdm
from hyper import hp
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_metric
from data import InferenceDataset 
from utils import load_dataset, load_dataset_test
from generate_algo import Generate

def main(
    data_path: str,
    model_path: str,
    save_path: str,
    decode_algorithm: str, 
    model_checkpoint: str
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = load_dataset_test(data_path)
    # Define tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "en-XX"
    # If model is in family of t5, add prefix in input
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "translate English to Romanian: "
    else:
        prefix = ""
    test_dataset = InferenceDataset(test_data, prefix, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    sacrebleu = datasets.load_metric("sacrebleu")
    out_dict = {}
    idx = 0
    score = 0
    for inputs in tqdm(test_dataloader, desc="Generating:"):
        datas = {key: value.to(device) for key,value in inputs.items()} 
        output = Generate(datas, tokenizer, model, decode_algorithm) 
        out_dict[str(idx)] = [
            test_data[idx][0],
            test_data[idx][1],
            output[0]
        ]
        results = sacrebleu.compute(predictions=output, references=[[test_data[idx][0]]])
        score += round(results["score"], 1)
        idx += 1
    print(f'BLUE score: {score/len(test_dataloader)}')
    with open(save_path, 'w') as fp:
        json.dump(out_dict, fp, indent=6) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--decode_algorithm", type=str, default="beam-search")
    parser.add_argument("--model_checkpoint", type=str, default="Helsinki-NLP/opus-mt-zh-en")
    args = parser.parse_args()
    main(**vars(args))
