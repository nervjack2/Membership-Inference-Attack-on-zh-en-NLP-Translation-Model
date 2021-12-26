

def load_data(path):
    with open(path, 'r') as fp:
        lines = [line.strip('').split(' ') for line in fp]
    return lines

def load_tgt_idx(path):
    with open(path, 'r') as fp:
        x = fp.readline()
        return list(map(int, x.strip().split(' ')))