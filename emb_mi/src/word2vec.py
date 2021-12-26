import numpy as np
import multiprocessing
from tqdm import tqdm
from gensim.models import word2vec

def train_word2vec(x, min_count, vector_size):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, vector_size=80, window=5, 
        min_count=min_count, workers=multiprocessing.cpu_count(), epochs=20, sg=1)
    return model

def train_multiple_word2vec(p_idx, n_idx, data, min_count, vector_size):
    p_model, n_model = [],[]
    for idx in tqdm(p_idx, desc='Train positive shadow model:'):
        sample_data = [data[x] for x in idx]
        model = train_word2vec(sample_data, min_count, vector_size)
        p_model.append(model)
    for idx in tqdm(n_idx, desc='Train negative shadow model:'):
        sample_data = [data[x] for x in idx]
        model = train_word2vec(sample_data, min_count, vector_size)
        n_model.append(model)
    return p_model, n_model

def extract_emb(p_wav2vec, n_wav2vec, tgt_idx, data, vector_size):
    p_emb, n_emb = [],[]
    tgt_data = [data[x] for x in tgt_idx]
    tgt_data = [word for email in tgt_data for word in email]
    unk_vec = np.random.uniform(0,1,vector_size)
    for model in p_wav2vec:
        emb_per_model = []
        for word in tgt_data:
            try:
                emb = model.wv.get_vector(word)
            except:
                emb = unk_vec
            emb_per_model.append(emb)
        p_emb.append(emb_per_model)
    for model in n_wav2vec:
        emb_per_model = []
        for word in tgt_data:
            try:
                emb = model.wv.get_vector(word)
            except:
                emb = unk_vec
            emb_per_model.append(emb)
        n_emb.append(emb_per_model)
    return p_emb, n_emb, unk_vec