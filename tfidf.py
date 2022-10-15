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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from lightgbm import LGBMClassifier
import pickle
import gensim
from gensim.models import Word2Vec

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
    map_dict = {"Food and Groceries": 0, "Medical and Healthcare": 1,"Education":2,"Lifestyle and Entertainment":3,"Travel & Transportation":4,"Clothing":5}#,"Eye":6,"Shoe":7}#"Housing and Utilities":3
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
            test_size=0.30,
            random_state=60,
            stratify=data["Category"],
        )
    return xtrain,xtest,ytrain,ytest

def word_vec(xtrain,xtest):
    print("[+] Vectorizing...")
    w2v = Word2Vec(data, min_count = 1, vector_size = 100,window = 5, sg = 1)
    train_df = pd.DataFrame(
                tfidf.transform(xtrain).toarray(), columns=tfidf.vocabulary_
            )

    test_df = pd.DataFrame(
                tfidf.transform(xtest).toarray(), columns=tfidf.vocabulary_
            )

    # pickle.dump(tfidf, open("tfidf.pickle", "wb"))
    return tfidf,train_df, test_df
#modeling    
def tfidf(xtrain,xtest):
    sleep(2)
    print("[+] Vectorizing...")
    tfidf = TfidfVectorizer(min_df = 5,sublinear_tf=True ,lowercase=True,max_features=300, stop_words="english")
    tfidf = tfidf.fit(xtrain)
    train_df = pd.DataFrame(
                tfidf.transform(xtrain).toarray(), columns=tfidf.vocabulary_
            )

    test_df = pd.DataFrame(
                tfidf.transform(xtest).toarray(), columns=tfidf.vocabulary_
            )

    # pickle.dump(tfidf, open("tfidf.pickle", "wb"))
    return tfidf,train_df, test_df

def fit_model(model,train_df,test_df,y_train,ytest):
    sleep(2)
    print("[+] Fitting Model...")
    model.fit(train_df, y_train)
    preds = model.predict(test_df)
    print(model)
    print(classification_report(ytest, preds))
    return model

def tune_hyperparameters(space):

    clf= XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( train_df, y_train), ( test_df, ytest)]
    
    clf.fit(train_df, y_train,
            eval_set=evaluation, eval_metric="mlogloss",verbose=False)
    pred = clf.predict(test_df)
    accuracy = metrics.accuracy_score(ytest, pred)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }



if __name__=="__main__":
    file_path = set_paths()
    data = read_data(file_path)
    data = preprocess(data)
    xtrain,xtest, y_train, ytest = training_utils(data)
    fitted_tfidf, train_df,test_df = tfidf(xtrain,xtest)
    st_x= StandardScaler()    
    train_df = st_x.fit_transform(train_df)    
    test_df = st_x.transform(test_df)    
    
    # model = KNeighborsClassifier(n_neighbors=63)
    # knn = fit_model(model,train_df,test_df,y_train,ytest)
   
    #model = DecisionTreeClassifier()
    #dt = fit_model(model,train_df,test_df,y_train,ytest)
   
    #model = RandomForestClassifier(n_estimators= 300, criterion="entropy")
    # rf = fit_model(model,train_df,test_df,y_train,ytest)

    # final_model = VotingClassifier(estimators=[('dt',dt),('rf',rf)])
    # vote = fit_model(final_model,train_df,test_df,y_train,ytest)
    
    
#    model = XGBClassifier(booster = 'dart')
#    xgb = fit_model(model,train_df,test_df,y_train,ytest)
    
    # model  = XGBClassifier()
    # xgb = fit_model(model,train_df,test_df,y_train,ytest)
    a=0
#    for row in test_df:
#        print(xgb.predict_proba(row.reshape(1,-1)))
#        print(xgb.predict(row.reshape(1,-1)))
#        a+=1
#        if a ==10:
#            break
    
#    pickle.dump(fitted_tfidf,open("tfidf2.pickle",'wb'))
    w = ["Arun Clothes","Pooja Medical","ABC Tours and Travels", "Dev Super Mart"]
    w1 = []
    for i in w:
         w1.append(fitted_tfidf.transform([i]))
    print(w1)
    
    w2 = []
    for i in w:
        w2.append(fitted_tfidf.transform([i]))
    print(w2)


    from sklearn.linear_model import LogisticRegression
    #multi_class="multinomial"
    model = LogisticRegression(solver='saga', max_iter=300,multi_class="multinomial",warm_start=True)
    lr = fit_model(model,train_df,test_df,y_train,ytest)
    print("LR")
    for i in range(len(w1)):
         print(w[i])
         print(lr.predict(w1[i]))
         print(lr.predict_proba(w1[i]))

    #model = KNeighborsClassifier(n_neighbors=75)
    #knn = fit_model(model,train_df,test_df,y_train,ytest)
    #print("KNN")
    #for i in range(len(w1)):
    #     print(w[i])
    #     print(knn.predict(w1[i]))
    #     print(knn.predict_proba(w1[i]))

    #print("DT")
    #for i in range(len(w1)):
    #     print(w[i])
    #     print(dt.predict(w1[i]))
    #     print(dt.predict_proba(w1[i]))
    filename = 'final_lr1.sav'


 #   pickle.dump(lr, open(filename, 'wb'))
    # model = XGBClassifier(colsample_bytree=0.6059329304964837,
    #                         gamma=2.361923398781385,
    #                         max_depth=12,
    #                         min_child_weight=6,
    #                         reg_alpha=41.0,
    #                         reg_lambda=0.00474534836744336,
    #                         booster = 'dart'
    #                         )

    # xgb = fit_model(model,train_df,test_df,y_train,ytest)

   
    # final_model = VotingClassifier(estimators=[('dt',dt),('knn',knn),('lr',lr)])
    # final_model = fit_model(final_model,train_df,test_df,y_train,ytest)
    # print("Final")
    # for i in range(len(w1)):
    #     print(w[i])
    #     print(dt.predict_proba(w1[i]))

    # trials = Trials()
    # space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
    #     'gamma': hp.uniform ('gamma', 1,9),
    #     'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    #     'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    #     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
    #     'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    #     'n_estimators': 180,
    #     'seed': 0
    # }

    # best_hyperparams = fmin(fn = tune_hyperparameters,
    #                         space = space,
    #                         algo = tpe.suggest,
    #                         max_evals = 100,
    #                         trials = trials)

    # with open(r'hyperparams.txt', 'w') as fp:
    #     for k,v in best_hyperparams.items():
    #         # write each item on a new line
    #         fp.write(f"{k}={v},\n")
    #     print('Done')
