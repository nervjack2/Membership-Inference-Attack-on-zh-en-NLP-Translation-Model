from opencc import OpenCC
from os.path import join


c = OpenCC('s2t')

def load_UN_dataset(data_dir):
    en_data_path = join(data_dir, 'UNv1.0.en-zh.en')
    zh_data_path = join(data_dir, 'UNv1.0.en-zh.zh')
    with open(en_data_path, 'r') as fp:
        en_data_list = []
        count = 0
        for x in fp:
            print(f'UN en:{900000}/{count+1}', end='\r')
            en_data_list.append(x.strip())
            count += 1 
            if count == 900000:
                break 
    with open(zh_data_path, 'r') as fp:
        zh_data_list = []
        count = 0
        for x in fp:
            print(f'UN zh:{900000}/{count+1}', end='\r')
            zh_data_list.append(c.convert(x.strip()))
            count += 1 
            if count == 900000:
                break 
    print(f'UN data example:\n{en_data_list[12]}\n{zh_data_list[12]}')
    print(len(en_data_list))
    return [[zh_x, en_x] for zh_x, en_x in zip(zh_data_list, en_data_list)]

def load_News_dataset(data_dir):
    en_data_path = join(data_dir, 'News-Commentary.en-zh.en')
    zh_data_path = join(data_dir, 'News-Commentary.en-zh.zh')
    with open(en_data_path, 'r') as fp:
        en_data_list = []
        count = 0
        for x in fp:
            if len(x.strip()) == 0:
                continue
            print(f'News en:{100000}/{count+1}', end='\r')
            en_data_list.append(x.strip())
            count += 1 
            if count == 100000:
                break 
    with open(zh_data_path, 'r') as fp:
        zh_data_list = []
        count = 0
        for x in fp:
            if len(x.strip()) == 0:
                continue
            print(f'News zh:{100000}/{count+1}', end='\r')
            zh_data_list.append(c.convert(x.strip()))
            count += 1 
            if count == 100000:
                break 
    print(f'News data example:\n{en_data_list[12]}\n{zh_data_list[12]}')
    print(len(en_data_list))
    return [[zh_x, en_x] for zh_x, en_x in zip(zh_data_list, en_data_list)]

def load_wiki_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = []
        count = 0 
        for x in fp:
            if len(x.strip().split('\t')) != 2:
                continue 
            print(f'wiki:{15000}/{count+1}', end='\r')
            data.append(x.strip().split('\t'))
            count += 1
            if count == 15000:
                break 

    en_data_list = [d[1] for d in data]
    zh_data_list = [c.convert(d[0]) for d in data]
    print(f'wiki data example:\n{en_data_list[12]}\n{zh_data_list[12]}')
    print(len(en_data_list))
    return [[zh_x, en_x] for zh_x, en_x in zip(zh_data_list, en_data_list)]

def load_dataset(data_path, dataset_name):
    if dataset_name == 'UN':
        data = load_UN_dataset(data_path)
    elif dataset_name == 'News':
        data = load_News_dataset(data_path)
    elif dataset_name == 'wiki':
        data = load_wiki_dataset(data_path)
    else:
        raise f'Dataset name {dataset_name} not found.'
    return data 
