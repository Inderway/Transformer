# -*- coding=utf8 -*-
# Aug 13, 2022
# For testing codes
# Created by Wei Yin

import torchtext.datasets as datasets
from torchtext.data.utils import get_tokenizer
import io
import spacy

# train_path=("data/train/train.de","data/train/train.en")
# val_path=("data/val/val.de","data/val/val.en")
# test_path=("data/test/test_2016_flickr.de","data/test/test_2016_flickr.en")

def getData(idx):
    train_path=("data/train/train.de","data/train/train.en")
    val_path=("data/val/val.de","data/val/val.en")
    test_path=("data/test/test_2016_flickr.de","data/test/test_2016_flickr.en")
    def data_process(filepath):
        lan_iter=iter(io.open(filepath[idx],encoding='utf8'))
        # a list that contains tuples of sentence pair
        data=[]
        for sentence in lan_iter:
            data.append(sentence.rstrip("\n"))
        return data
    train=data_process(train_path)
    val=data_process(val_path)
    test=data_process(test_path)
    return train,val,test


train,val,test=getData(1)

def tokenize(text, tokenizer):
    # 返回分词结果列表
    return [tok.text for tok in tokenizer.tokenizer(text)]

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def yield_tokens(data_iter, tokenizer):
    for from_to_tuple in data_iter[0:10]:
        res=tokenizer(from_to_tuple)
        print(from_to_tuple)
        print(res)
        
yield_tokens(test,spacy_en)

