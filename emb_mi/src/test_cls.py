import os
import numpy as np 
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from sklearn.linear_model import Lasso
from gensim.models import word2vec
from load_data import load_data, load_tgt_idx
from utils import sample_shadow_data, calculate_emb_dis
from word2vec import train_multiple_word2vec, extract_emb


def main(
    data_path: str,
    tgt_idx_path: str,
    tgt_model_dir: str,
    cls_model_path: str,
    unk_vec_path: str,
    tgt_size: int 
):
    data = load_data(data_path)
    data = data[:len(data)//2]
    tgt_idx = load_tgt_idx(tgt_idx_path)[:tgt_size]
    tgt_data = [data[x] for x in tgt_idx]
    tgt_data = [word for email in tgt_data for word in email]
    cls_model = pickle.load(open(cls_model_path,'rb'))
    coef = cls_model.coef_
    query_index = np.nonzero((coef[:-1] != 0) * (coef[1:] != 0))[0]
    unk_vec = np.load(unk_vec_path)
    correct, total = 0, 0
    for emb_model_name in tqdm(os.listdir(tgt_model_dir), desc='Evaluation:'):
        b = 1 if emb_model_name[0] == 'p' else -1
        emb_model_path = os.path.join(tgt_model_dir, emb_model_name)
        tgt_emb_model = word2vec.Word2Vec.load(emb_model_path)
        word_emb = []
        for i, word in enumerate(tgt_data):
            if i not in query_index:
                continue
            try:
                emb = tgt_emb_model.wv.get_vector(word)
            except:
                emb = unk_vec
            word_emb.append(emb)
        word_emb_dis = np.zeros((1,len(tgt_data)-1))
        pre_w_emb = word_emb[0]
        for i, (q_idx, w_emb) in enumerate(zip(query_index[:-1],word_emb[1:])):
            emb_dis = (((pre_w_emb - w_emb)**2).sum())**(1/2) # Euclidean distance
            pre_w_emb = w_emb
            word_emb_dis[0,q_idx] = emb_dis
        y = cls_model.predict(word_emb_dis)
        total += 1 
        if y*b > 0:
            correct += 1 
    print('Accuracy:', correct/total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tgt_idx_path', type=str)
    parser.add_argument('--tgt_model_dir', type=str)
    parser.add_argument('--cls_model_path', type=str)
    parser.add_argument('--unk_vec_path', type=str)
    parser.add_argument('--tgt_size', type=int, default=50)
    args = parser.parse_args()
    main(**vars(args))
