import os 
import numpy as np 
import pandas as pd
import argparse
import pickle
from load_data import load_data, load_tgt_idx
from utils import sample_target_data, calculate_emb_dis
from word2vec import train_multiple_word2vec, extract_emb


def main(
    data_path: str,
    tgt_idx_path: str,
    save_model_dir: str,
    n_tgt_model: int,
    tgt_size: int,
    train_size: int,
    w_min: int
):
    data = load_data(data_path)
    tgt_idx = load_tgt_idx(tgt_idx_path)[:tgt_size]
    p_idx, n_idx = sample_target_data(tgt_idx, train_size, len(data), n_tgt_model)
    p_wav2vec, n_wav2vec = train_multiple_word2vec(p_idx, n_idx, data, w_min)
    for i, model in enumerate(p_wav2vec):
        model.save(os.path.join(save_model_dir, f'p_word2vec_{i}.mdl'))
    for i, model in enumerate(n_wav2vec):
        model.save(os.path.join(save_model_dir, f'n_word2vec_{i}.mdl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tgt_idx_path', type=str)
    parser.add_argument('--save_model_dir', type=str)
    parser.add_argument('--n_tgt_model', type=int, default=50)
    parser.add_argument('--tgt_size', type=int, default=50)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--w_min', type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
