# %%
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn import metrics
from tqdm import tqdm


from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(902022)



# %%
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
ids_in = tokenizer.encode('in </s>')
ids_out = tokenizer.encode('out </s>')

class AliceDataset(Dataset):
  def __init__(self, tokenizer, data_dir, max_len=512):
    self.in_file_path = os.path.join(data_dir, 'A_in.json')
    self.out_file_path = os.path.join(data_dir, 'A_out.json')
    
    self.in_files = json.load(open(self.in_file_path))
    self.out_files = json.load(open(self.out_file_path))
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._buil_examples_from_files(self.in_files, 'in')
    self._buil_examples_from_files(self.out_files, 'out')
  
  def _buil_examples_from_files(self, files, sentiment):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for k, v in files.items():

      line = v[0] + v[1] + v[2]
      line = REPLACE_NO_SPACE.sub("", line) 
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line + ' </s>'

      target = sentiment + " </s>"

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)

# %%
dataset = AliceDataset(tokenizer, '', 'test',  max_len=512)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model = MT5ForConditionalGeneration.from_pretrained()
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)

print(metrics.classification_report(targets, outputs))