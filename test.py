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
st.subheader("Upload a txt file with each line containing a brand, we'll tell you their categories")

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
filename = 'final_lr1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

with open('tfidf2.pickle','rb') as to_read:
   fitted_tfidf = pickle.load(to_read)

map_dict = {0:"Food and Groceries", 1:"Medical and Healthcare",2:"Education",3:"Lifestyle and Entertainment",4:"Travel & Transportation",5:"Clothing"}

def predict_model(brand):
    bo,ind = is_str_in(brand,brands)
    if bo:
        out = categories[ind]
 
    else:
        w = fitted_tfidf.transform([brand])
                    # print(w)

        pred = loaded_model.predict(w)
        out = map_dict[pred[0]]
    return out
                # print(loaded_model.predict(w))
                #out = categories[out]

import time
# brand = st.text_input("Enter the name of the brand")
#     submit = st.form_submit_button('Submit')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    uploaded_file = uploaded_file.getvalue().decode('utf-8').splitlines()
        # st.write(uploaded_file)

# print the list
    #print(content_list)

# remove new line characters
    brand_list = [x.strip() for x in uploaded_file]
        #st.write(" ".join(content_list))
    st.subheader("Output File")
    with st.spinner(text="This may take a moment..."):
                time.sleep(2)
                out_list = []
                for brand in brand_list:
                    out_list.append(brand+" -> "+predict_model(brand))
                
                # bo,ind = is_str_in(brand,brands)
                # if bo:
                #     out = categories[ind]
    
                # else:
                #     a,out1,_,_=find_closest_match(brand,brands)
                #     w = fitted_tfidf.transform([brand])
                #         # print(w)

                #     pred = loaded_model.predict(w)
                #     out = map_dict[pred[0]]
                #     out = "Normal String matching:"+str(categories[out1])+"\n"+" Model:"+out
                    # print(loaded_model.predict(w))
                    #out = categories[out]

    out = "\n".join(out_list)
    st.download_button('Download Outputs', out) 
    
#'''

#while True:
    #w =input()
    #if w == 'b':
    #    break
#'''
