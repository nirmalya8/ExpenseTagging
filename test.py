#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tfidf import set_paths, read_data, preprocess, training_utils, tfidf


#tfidf = TfidfVectorizer(max_features=300, stop_words="english")

filename = 'finalized_model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

with open('tfidf.pickle','rb') as to_read:
    fitted_tfidf = pickle.load(to_read)

w = "Arun Optical"
w = fitted_tfidf.transform([w])
print(w)

pred = loaded_model.predict_proba(w)
print(pred)
