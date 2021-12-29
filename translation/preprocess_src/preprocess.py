"""
    Usage: Translate Simplified Chinese label to Traditional Chinese label, 
                and merge parallel data into one json file. Split three datset into 
                Alice's training set, out probe, in probe, bob training set and ood probe.
"""
import os 
import argparse
import json
from os.path import join
from load_dataset import load_dataset

def main(
    UN_data_dir: str,
    News_data_dir: str,
    wiki_data_path: str,  
    output_dir: str
):   
    os.makedirs(output_dir, exist_ok=True)
    UN_data = load_dataset(UN_data_dir, 'UN')
    News_data = load_dataset(News_data_dir, 'News')
    wiki_data = load_dataset(wiki_data_path, 'wiki')
    # Alice training data
    A_train_dict = {}
    idx = 0
    for data in UN_data[:800000]:
        A_train_dict[idx] = data 
        idx += 1 
    for data in News_data[:80000]:
        A_train_dict[idx] = data 
        idx += 1 
    A_train_path = join(output_dir, 'A_train.json')
    with open(A_train_path, 'w') as fp:
        json.dump(A_train_dict, fp, indent=6)
    # Alice Evaluation data
    A_eval_dict = {}
    idx = 0
    for data in UN_data[800000:810000]:
        A_eval_dict[idx] = data 
        idx += 1 
    for data in News_data[80000:81000]:
        A_eval_dict[idx] = data 
        idx += 1 
    A_eval_path = join(output_dir, 'A_eval.json')
    with open(A_eval_path, 'w') as fp:
        json.dump(A_eval_dict, fp, indent=6)
    # Alice out probe 
    A_out_probe_dict = {}
    idx = 0
    for data in UN_data[810000:820000]:
        A_out_probe_dict[idx] = data 
        idx += 1 
    for data in News_data[81000:82000]:
        A_out_probe_dict[idx] = data 
        idx += 1 
    A_out_probe_path = join(output_dir, 'A_out.json')
    with open(A_out_probe_path, 'w') as fp:
        json.dump(A_out_probe_dict, fp, indent=6)
    # Alice in probe 
    A_in_probe_dict = {}
    idx = 0
    for data in UN_data[:10000]:
        A_in_probe_dict[idx] = data 
        idx += 1 
    for data in News_data[:1000]:
        A_in_probe_dict[idx] = data 
        idx += 1 
    A_in_probe_path = join(output_dir, 'A_in.json')
    with open(A_in_probe_path, 'w') as fp:
        json.dump(A_in_probe_dict, fp, indent=6)
    # Bob training data
    B_train_dict = {}
    idx = 0
    for data in UN_data[10000:400000]:
        B_train_dict[idx] = data 
        idx += 1 
    B_train_path = join(output_dir, 'B_train.json')
    with open(B_train_path, 'w') as fp:
        json.dump(B_train_dict, fp, indent=6)
    # Bob in probe
    B_in_probe_dict = {}
    idx = 0
    for data in UN_data[10000:400000]:
        B_in_probe_dict[idx] = data 
        idx += 1 
    B_in_probe_path = join(output_dir, 'B_in.json')
    with open(B_in_probe_path, 'w') as fp:
        json.dump(B_in_probe_dict, fp, indent=6)
    # Bob out probe 
    B_out_probe_dict = {}
    idx = 0
    for data in UN_data[410000:800000]:
        B_out_probe_dict[idx] = data 
        idx += 1
    B_out_probe_path = join(output_dir, 'B_out.json')
    with open(B_out_probe_path, 'w') as fp:
        json.dump(B_out_probe_dict, fp, indent=6)
    # Alice out of domain probe 
    A_ood_probe_dict = {}
    idx = 0
    for data in wiki_data[:1000]:
        A_ood_probe_dict [idx] = data 
        idx += 1 
    A_ood_probe_path = join(output_dir, 'A_ood.json')
    with open(A_ood_probe_path, 'w') as fp:
        json.dump(A_ood_probe_dict, fp, indent=6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--UN_data_dir', type=str)
    parser.add_argument('--News_data_dir', type=str)
    parser.add_argument('--wiki_data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    main(**vars(args))
