import torch
from transformers import(MinLengthLogitsProcessor,
                        HammingDiversityLogitsProcessor,
                        BeamSearchScorer,
                        LogitsProcessorList)

def Generate(datas, tokenizer, model, algorithm):
    if algorithm == 'beam-search':
        output = model.generate(
            **datas,
            max_length=70,
            min_length=20,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    elif algorithm == 'greedy':
        output = model.generate(
            **datas, 
            max_length=70,
            min_length=20,
        )
    elif algorithm == 'temperature':
        output = model.generate(
            **datas,
            max_length=70,
            min_length=20,
            do_sample=True,
            top_k=0,
            temperature=0.7
        )
    elif algorithm == 'top-k':
        output = model.generate(
            **datas,
            max_length=70,
            min_length=20,
            do_sample=True,
            top_k=50,
        )
    elif algorithm == 'top-p':
        output = model.generate(
            **datas,
            max_length=70,
            min_length=20,
            do_sample=True,
            top_p=0.9,
            top_k=0,
        )
    else:
        print(f'Algorithm {algorithm} not supported.')
        exit(0)
    with tokenizer.as_target_tokenizer():
        return [tokenizer.decode(x[1:], skip_special_tokens=True) for x in output]