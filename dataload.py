import conllu
import os
import pandas as pd
import numpy as np
import re
from os import listdir
import tensorflow
from os.path import isfile, isdir, join

def load_raw_conllu(train_data_path=None, val_data_path=None,test_data_path=None):
    filepaths = []
    if train_data_path != None:
        filepaths += [train_data_path]
    if val_data_path != None:
        filepaths += [val_data_path]
    if test_data_path != None:
        filepaths += [test_data_path]
    data_set_list = []
    for path in filepaths:
        print("Processing {}".format(path))
        with open(path, "r") as f:
            raw_data = f.read()
        data_set_list.append(conllu.parse(raw_data))
        
    return data_set_list

def convert2df(raw_data_list):
    data_frame_list = []
    for idx, data in enumerate(raw_data_list):
        sentence_dfs = []
        cols = data[0][0].keys()
        for i in range(len(data)):
            words = []
            for j in range(len(data[i])):
                words.append(data[i][j].values())
            sentence_dfs.append(pd.DataFrame(words, columns=cols).set_index('id'))
        data_frame_list.append(sentence_dfs)
    return data_frame_list