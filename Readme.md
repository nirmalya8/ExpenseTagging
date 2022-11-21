# Expense Tagging

A live demo of this project is available on [HuggingFace(Click Here)](https://huggingface.co/spaces/nirmalya8/expense_tagging).

The Readme is divided into the following sections: <ul>
    <li> [Problem Statement](#problem-statement) </li>
    <li> [Proposed Approach](#proposed-approach) </li>
    <li> [Running The Code](#running-the-code) </li>
    <li> [File System](#file-system)</li>

</ul>

## Problem Statement
---
Given the name of a brand, a company, or a shop predict which of the following categories it belongs to: Food and Groceries, Medical and Healthcare, Education, Lifestyle and Entertainment, and Travel & Transportation.

Input : Name of a brand  
Output: Category

## Proposed Approach
---
The main approach to this problem lies in the encoding of the words to numbers. When we look at the name of a store say `XYZ Grocery Shop`, we immediately look at the word `Grocery` and go it must be a Food and Groceries store. Similarly, `XYZ Pharmacy` would almost always be a Medicine store. 

In the word space, names belonging to a certain category would always end up in a cluster, far from any other category. Or, in theory, that should be the case.

What we, however, can't distinguish is brand names, such as Levi's or Apollo, if we don't know their category only. So, the proposed approach consists of two parts.

One is normal string matching for the most popular brands whose names might not give away their category. The other part is a Machine Learning model, which would tell us the category of the brand.

On training, Logistic Regression gave an overall test accuracy of 81%. We chose to go forward with it. 

``` [+] Ensembling techniques i.e. Random Forests, Voting Classifier, XGBoost etc did not seem to work on test data when they were fed individually. So, even though XGBoost gave an accuracy of 84%, we didn't go with it. ```


## Running The Code
---
The code works on any system with or above Python 3.9.

Clone the repository and move to it in the terminal using the `cd` command.

Let's install the dependencies and libraries used. Although not many, these are contained in the `requirements.txt` file and need to be installed via the terminal using pip.

```
pip install -r requirements.txt
```

Now, to run the web app on streamlit, run:
```
cd src
streamlit run test.py
```
## File System
---
This is the file structure. Details on each are given below.
<pre>
├── __pycache__  
├── Readme.md  
├── requirements.txt  
└── src  
    ├── brands.json  
    ├── check.py  
    ├── Data  
    │   └── Consolidated Expense Tagging.xlsx  
    ├── get_brands.py  
    ├── Misc  
    │   ├── brands.txt  
    │   ├── categories.txt  
    │   ├── hyperparams.txt 
    │   └── test_file.txt 
    ├── Models 
    │   ├── finalized_model1.sav 
    │   ├── finalized_model.sav 
    │   ├── final_lr1.sav 
    │   ├── final_lr.sav 
    │   ├── tfidf2.pickle 
    │   └── tfidf.pickle 
    ├── __pycache__ 
    ├── test.py
    └── tfidf.py
    </pre>
This is the `Readme.md` file. The `requirements.txt` file contains all libraries which need to be installed. The `src` folder contains all the code, Data and Models. The `Data` folder contains the dataset in a `.csv` file. The `Models` folder contains all the already trained models for both vectorization and Prediction. `Misc` contains a few miscelleneous files such as names of popular brands, their categories, a hyperparameter test and a test file for the streamlit application.

`tfidf.py` is the file in which everything is done, from importing the data, to cleaning it to building and saving the model. `brands.json` contains the names of popular brands with their categories as key:value pairs. `test.py` integrates string matching and the Logistic Regression Model and deploys it in a streamlit web app. 