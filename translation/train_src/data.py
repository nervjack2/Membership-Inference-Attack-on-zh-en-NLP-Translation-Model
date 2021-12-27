import torch 
from hyper import hp
from torch.utils.data import Dataset

class TraningDataset(Dataset):
    def __init__(self, data, prefix, tokenizer):
        self.inputs = [prefix+x[0] for x in data]
        self.targets = [prefix+x[1] for x in data]
        self.model_inputs = tokenizer(self.inputs, max_length=hp.max_input_length, 
                                truncation=True, padding=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            self.labels = tokenizer(self.targets, max_length=hp.max_target_length, 
                                truncation=True, padding=True)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_data = {k: torch.LongTensor(v[idx]) for k,v in self.model_inputs.items()}   
        input_data['labels'] = torch.LongTensor(self.labels['input_ids'][idx])
        return input_data

class InferenceDataset(Dataset):
    def __init__(self, data, prefix, tokenizer):
        self.inputs = [prefix+x[0] for x in data]
        self.targets = [prefix+x[1] for x in data]
        self.model_inputs = tokenizer(self.inputs, max_length=hp.max_input_length, 
                                truncation=True, padding=True)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_data = {k: torch.LongTensor(v[idx]) for k,v in self.model_inputs.items()}   
        return input_data
