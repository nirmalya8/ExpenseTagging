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
from check import is_str_in
#tfidf = TfidfVectorizer(max_features=300, stop_words="english")
import streamlit as st

st.title("Expense Tagging")
st.subheader("Type in the name of a brand, we'll tell you its category")

file_name = "brands.json"
with open(file_name,'r') as f:
    data = json.load(f)
brands = []
categories = []
for k in data.keys():
    brands.append(data[k]["name"])
    categories.append(data[k]["category"])

#i = input()
#print(len(brands), len(categories))
#bo,ind = is_str_in(i,brands)
#a,b,_,_ = find_closest_match(i,brands)
#print(a,b,brands[b])
#if bo:
#    print(categories[ind])
#print(categories[b])

with st.form("form1",clear_on_submit=False):
    brand = st.text_input("Enter the name of the brand")
    submit = st.form_submit_button('Submit')
    if submit:
        st.subheader("Output Text")
        with st.spinner(text="This may take a moment..."):
            bo,ind = is_str_in(brand,brands)
            if bo:
                out = categories[ind]

            else:
                a,out,_,_=find_closest_match(brand,brands)
                out = categories[out]
        st.write(out)
    
#'''
#filename = 'final_lr1.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

#with open('tfidf2.pickle','rb') as to_read:
 #   fitted_tfidf = pickle.load(to_read)

#while True:
    #w =input()
    #if w == 'b':
    #    break
    #w = fitted_tfidf.transform([w])
    #print(w)

    #pred = loaded_model.predict_proba(w)
    #print(pred)
    #print(loaded_model.predict(w))'''
