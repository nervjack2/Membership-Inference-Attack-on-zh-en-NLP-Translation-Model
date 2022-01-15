# Chinese-to-English NLP Translation

## Dataset 
- A_train.json: 800000 from UN Parallel Corpus V1.0, 8000 from News Commentary v16
- B_train.json: 390000 from UN Parallel Corpus V1.0, B_train is a subset of A_train 
- A_in.json, A_out.json: 10000 from UN Parallel Corpus V1.0, 1000 from News Commentary v16
- A_ood.json: 1000 from wikititiles v1

## Script Usage 
- download.sh 
    - Download preprocessed data, target model and shadow model
- inference.sh $1 $2 $3 $4
    - $1: data json file path 
    - $2: model checkpoint path 
    - $3: save result path
        - result will be saved in a json file with [f, e, e'] format
    - $4: model_checkpoint 
        - Helsinki-NLP/opus-mt-zh-en or liam168/trans-opus-mt-zh-en

## Model Structure and Performance 
Alice's target model: 
- AutoModelForSeq2SeqLM with checkpoint "Helsinki-NLP/opus-mt-zh-en"
- GLUE score: 36.04 

Bob's shadow model:
- AutoModelForSeq2SeqLM with checkpoint "liam168/trans-opus-mt-zh-en"
- GLUE score: 34.92
- Using 390000 training data

Bob's shadow model (smaller version):
- AutoModelForSeq2SeqLM with checkpoint "liam168/trans-opus-mt-zh-en"
- GLUE score: 32.19
- Using 40000 training data

## Preprocess 
Install opencc from this repository for Simplified Chinese to Traditional Chinese translation 
```
https://github.com/yichen0831/opencc-python
```
Or you could also download data which have already been preprocessed by download_dataset.sh  
