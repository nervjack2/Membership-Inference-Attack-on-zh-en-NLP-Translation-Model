# Chinese-to-English NLP Translation
## Preprocess 
Install opencc from this repository for Simplified Chinese to Traditional Chinese translation 
```
https://github.com/yichen0831/opencc-python
```
Or you could also download data which have already been preprocessed from this Dropbox site:
```
wget 
```  
## Dataset 
- A_train.json: 800000 from UN Parallel Corpus V1.0, 8000 from News Commentary v16
- B_train.json: 390000 from UN Parallel Corpus V1.0, B_train is a subset of A_train 
- A_in.json, A_out.json: 10000 from UN Parallel Corpus V1.0, 1000 from News Commentary v16
- A_ood.json: 1000 from wikititiles v1

## Script Usage 
- inference.sh $1 $2 $3
    - $1: data json file path 
    - $2: model checkpoint path 
    - $3: save result path
        - result will be saved in a json file, with [f, e, e'] format