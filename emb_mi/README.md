# Membership Inference on Word Embedding and Beyond
## Description
This repository is the reproduction of paper https://arxiv.org/abs/2106.11384
## Performance 
In the setting of the following parameters, got 50% classification accuracy:
```
Number of shadow model(Per class): 50
The size of target dataset: 50 
The total size of word2vec training data: 10000
min_count parameters of word2vec: 25
vector_size parameters of word2vec: 80
```
## Code Usage 
### src/train_cls.py
Training a classifier to classify whether the target data is used in the training phase of a word2vec embedding model.

Parameters:
--data_path : A .txt file, each line represents the content of an email 
--tgt_idx_path: A .txt file, contains the indexes of data seperated by white space in target data set. The index of a data is equivalent to the row of it in --data_path 
### src/train_tgt_model.py
Training target word2vec model 

Parameters:
--data_path : A .txt file, each line represents the content of an email 
--tgt_idx_path: A .txt file, contains the indexes of data seperated by white space in target data set. The index of a data is equivalent to the row of it in --data_path 
--n_tgt_model: Number of target model trained
### src/test_cls.py
Testing the accuracy of the classifier trained by src/train.py, using target word2vec models trained by src/train_tgt_model.py as evaluation metric

Parameters:
--data_path : A .txt file, each line represents the content of an email 
--tgt_idx_path: A .txt file, contains the indexes of data seperated by white space in target data set. The index of a data is equivalent to the row of it in --data_path 
--unk_vec_path: Unknown words vector embedding using in the training phase of LASSO classifier