import random
import numpy as np 
from tqdm import tqdm

def sample_shadow_data(tgt_idx, train_size, total_len, n_shadow_model):
    p_idx, n_idx = [], []
    sample_size = train_size-len(tgt_idx)
    # Construct a index list disjoint to tgt_idx 
    total_idx = list(range(total_len))
    for idx in tgt_idx:
        total_idx.remove(idx)
    # Sample positive set 
    for _ in tqdm(range(n_shadow_model), desc='Get positive index:'):
        random.shuffle(total_idx) 
        p_idx.append(total_idx[:sample_size].copy()+tgt_idx.copy())
    # Sample negative set 
    for _ in tqdm(range(n_shadow_model), desc='Get negative index:'):
        random.shuffle(total_idx)
        n_idx.append(total_idx[:train_size].copy())
        
    return p_idx, n_idx

def sample_target_data(tgt_idx, train_size, total_len, n_shadow_model):
    p_idx, n_idx = [], []
    sample_size = train_size-len(tgt_idx)
    total_idx = list(range(total_len-total_len//2))
    # Sample positive set 
    for _ in tqdm(range(n_shadow_model), desc='Get positive index:'):
        random.shuffle(total_idx) 
        p_idx.append(list(map(lambda x: x+total_len//2,total_idx[:sample_size].copy()))+tgt_idx.copy())
    # Sample negative set 
    for _ in tqdm(range(n_shadow_model), desc='Get negative index:'):
        random.shuffle(total_idx)
        n_idx.append(list(map(lambda x: x+total_len//2,total_idx[:train_size].copy())))
    return p_idx, n_idx

def calculate_emb_dis(p_emb, n_emb):
    emb_dis_list = []
    for emb in p_emb:
        dis_per_emb = []
        pre_w_emb = emb[0]
        for w_emb in emb[1:]:
            emb_dis = (((pre_w_emb - w_emb)**2).sum())**(1/2) # Euclidean distance
            pre_w_emb = w_emb
            dis_per_emb.append(emb_dis)
        emb_dis_list.append(np.array(dis_per_emb))
    for emb in n_emb:
        dis_per_emb = []
        pre_w_emb = emb[0]
        for w_emb in emb[1:]:
            emb_dis = (((pre_w_emb - w_emb)**2).sum())**(1/2) # Euclidean distance
            pre_w_emb = w_emb
            dis_per_emb.append(emb_dis)
        emb_dis_list.append(np.array(dis_per_emb))
    emb_dis_merge = np.stack(emb_dis_list)
    return emb_dis_merge