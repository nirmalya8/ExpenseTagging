#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tfidf import set_paths, read_data, preprocess, training_utils, tfidf
from check import find_closest_match

#tfidf = TfidfVectorizer(max_features=300, stop_words="english")

file_name = "brands.json"
with open(file_name,'r') as f:
    data = json.load(f)
brands = []
categories = []
for k in data.keys():
    brands.append(data[k]["name"])
    categories.append(data[k]["category"])

print(len(brands), len(categories))
a,b,_,_ = find_closest_match(input(),brands)
print(a,b,brands[b])
'''
filename = 'final_lr1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

with open('tfidf2.pickle','rb') as to_read:
    fitted_tfidf = pickle.load(to_read)

while True:
    w =input()
    if w == 'b':
        break
    w = fitted_tfidf.transform([w])
    print(w)

    pred = loaded_model.predict_proba(w)
    print(pred)
    print(loaded_model.predict(w))'''
