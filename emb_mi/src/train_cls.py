import os
import numpy as np 
import pandas as pd
import argparse
import pickle
from sklearn.linear_model import Lasso
from load_data import load_data, load_tgt_idx
from utils import sample_shadow_data, calculate_emb_dis
from word2vec import train_multiple_word2vec, extract_emb


def main(
    data_path: str,
    tgt_idx_path: str,
    save_model_dir: str,
    n_shadow_model: int,
    tgt_size: int,
    train_size: int,
    min_count: int,
    vector_size: int 
):
    data = load_data(data_path)
    data = data[:len(data)//2]
    tgt_idx = load_tgt_idx(tgt_idx_path)[:tgt_size]
    p_idx, n_idx = sample_shadow_data(tgt_idx, train_size, len(data), n_shadow_model)
    p_wav2vec, n_wav2vec = train_multiple_word2vec(p_idx, n_idx, data, min_count, vector_size)
    p_tgt_emb, n_tgt_emb, unk_vec = extract_emb(p_wav2vec, n_wav2vec, tgt_idx, data, vector_size)
    emb_dis = calculate_emb_dis(p_tgt_emb, n_tgt_emb)    
    label = np.array([1]*n_shadow_model+[-1]*n_shadow_model)
    model = Lasso(alpha=2/(n_shadow_model**(1/2)))
    model.fit(emb_dis, label)
    pickle.dump(model, open(os.path.join(save_model_dir,f'tar_cls_min_{min_count}.mdl'), 'wb'))
    np.save(os.path.join(save_model_dir, f'unk_vec_{min_count}.npy'), unk_vec)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tgt_idx_path', type=str)
    parser.add_argument('--save_model_dir', type=str)
    parser.add_argument('--n_shadow_model', type=int, default=50)
    parser.add_argument('--tgt_size', type=int, default=50)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--min_count', type=int, default=25)
    parser.add_argument('--vector_size', type=int, default=80)
    args = parser.parse_args()
    main(**vars(args))
