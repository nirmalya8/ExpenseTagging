#imports
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier 
from time import sleep
from sklearn.model_selection import KFold 
from sklearn import metrics

def set_paths():
    print("[+] Setting paths...")
    base_path = os.getcwd()
    data_path = os.path.join(base_path,'Data')
    models_path = os.path.join(base_path,'Models')
    file_path = os.path.join(data_path,'Consolidated Expense Tagging.xlsx')
    return file_path

def read_data(file_path):
    sleep(2)
    print("[+] Reading data...")
    xls = pd.ExcelFile(file_path)
    data = pd.read_excel(xls)
    print(data.head())
    print(data["Category"].value_counts())
    return data
#Preprocessing the data
def preprocess(data):
    sleep(2)
    print("[+] Preprocessing...")
    map_dict = {"Food and Groceries": 0, "Medical and Healthcare": 1,"Education":2,"Lifestyle and Entertainment":3,"Travel & Transportation":4}#"Housing and Utilities":3
    data["Category"] = data["Category"].map(map_dict)
    print(map_dict)
    return data
#train-test split
def training_utils(data):
    sleep(2)
    print("[+] Splitting data...")
    xtrain, xtest, ytrain, ytest = train_test_split(
            data['Name'],
            data["Category"],
            test_size=0.25,
            random_state=60,
            stratify=data["Category"],
        )
    return xtrain,xtest,ytrain,ytest

#modeling    
def tfidf(xtrain,xtest):
    sleep(2)
    print("[+] Vectorizing...")
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")

    train_df = pd.DataFrame(
                tfidf.fit_transform(xtrain).toarray(), columns=tfidf.vocabulary_
            )

    test_df = pd.DataFrame(
                tfidf.transform(xtest).toarray(), columns=tfidf.vocabulary_
            )
    return train_df, test_df

def fit_model(model,train_df,test_df,y_train,ytest):
    sleep(2)
    print("[+] Fitting Model...")
    model.fit(train_df, y_train)
    preds = model.predict(test_df)
    print(model)
    print(classification_report(ytest, preds))
    return model

if __name__=="__main__":
    file_path = set_paths()
    data = read_data(file_path)
    data = preprocess(data)
    xtrain,xtest, ytrain, ytest = training_utils(data)
    train_df,test_df = tfidf(xtrain,xtest)
    st_x= StandardScaler()    
    train_df = st_x.fit_transform(train_df)    
    test_df = st_x.transform(test_df)    
    
    model = KNeighborsClassifier(n_neighbors=63)
    knn = fit_model(model,train_df,test_df,ytrain,ytest)
    model = DecisionTreeClassifier()
    dt = fit_model(model,train_df,test_df,ytrain,ytest)
    model = RandomForestClassifier(n_estimators= 300, criterion="entropy")
    rf = fit_model(model,train_df,test_df,ytrain,ytest)

    final_model = VotingClassifier(estimators=[('dt',dt),('rf',rf)])
    vote = fit_model(final_model,train_df,test_df,ytrain,ytest)
    model = XGBClassifier()
    xgb = fit_model(model,train_df,test_df,ytrain,ytest)

    final_model = VotingClassifier(estimators=[('rf',rf),('xgb',xgb)])
    final_model = fit_model(final_model,train_df,test_df,ytrain,ytest)