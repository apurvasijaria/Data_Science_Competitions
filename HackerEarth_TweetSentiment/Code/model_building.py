# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:53:03 2020

@author: a.sijaria
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

##changing the working directory
os.chdir("C:/Users/<curr_directory>/sentiment analysis/code")


## loading the cleaned dataset
train = pd.read_csv("../dataset/train_cleaned.csv")
test = pd.read_csv("../dataset/test_cleaned.csv")

##analyizing cols
train.columns

##for the initial baseline model
#train_cls = ['retweet_count','cnt_words', 'ttl_wrds', 'clean_text_polarity','clean_text_subj'
#             ,'original_text_polarity','original_text_subj']

##final model variables
train_cls = ['retweet_count', 'cnt_words','ttl_wrds','clean_text_polarity','clean_text_subj']
test_cls = ['sentiment_class']

##separating traing and testing columns
df_train = train [train_cls]
df_test = train[test_cls]

##train test split 
X_train, X_test, y_train, y_test = train_test_split(df_train, df_test, test_size=0.25, random_state=7)

##------------------------------------------------------------------------
###linear regression
##model fit
reg = LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
reg.coef_

##predict 
y_pred = reg.predict(X_test)
y_adj = []
for v in y_pred:
    print(v)
    if v <= -0.033:
        y_adj.append(-1)
    elif v >= 0.033:
        y_adj.append(1)
    else:
        y_adj.append(0)
        
        
##accuracy measure
linear_accuracy = 100*f1_score(y_test, y_adj, average='weighted')
print(linear_accuracy)


##make test file iwth test predictions 
test_pred= test[train_cls]
pred_values = reg.predict(test_pred)
y_adj = []
div = 0.033
for v in pred_values:
    print(v)
    if v <= -1*div:
        y_adj.append(-1)
    elif v >= div:
        y_adj.append(1)
    else:
        y_adj.append(0)
id_v = test['id']  
df_final_linear = pd.DataFrame({'id':id_v,'sentiment_class':y_adj})    
df_final_linear.to_csv(''.join(["../submissions/linear_14_",str(div),".csv"]),index = False)

##------------------------------------------------------------------------
###logisticregression

## model fitting
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
logistict_pred = clf.predict_proba(X_test)
y_adj = []
for v in range(0,len(logistict_pred)):
    if logistict_pred[v,1] >= 0.3:
        y_adj.append(-1)
    elif logistict_pred[v,1] >= 0.3:
        y_adj.append(1)
    else:
        y_adj.append(0)
        
        
conf_m = confusion_matrix(y_test, logistict_pred)
report = classification_report(y_test, logistict_pred)

##make test file
test_pred= test[train_cls]
pred_values = clf.predict_proba(test_pred)
y_adj = []
for v in range(0,len(pred_values)):
    if pred_values[v,1] >= 0.3:
        y_adj.append(-1)
    elif pred_values[v,1] >= 0.3:
        y_adj.append(1)
    else:
        y_adj.append(0)

id_v = test['id']  
df_final_linear = pd.DataFrame({'id':id_v,'sentiment_class':y_adj})    
df_final_linear.to_csv("../submissions/log1.csv",index = False)

##accuracy
linear_accuracy = 100*f1_score(y_test, y_adj, average='weighted')
print(linear_accuracy)

##------------------------------------------------------------------------
##kmeans

## model fitting
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
kmeans.labels_

## predtion on test file
y_pred = kmeans.predict(X_test)
y_adj  = [w-1 for w in y_pred]
y_adj=[]
for v in pred_values:
    if v == 2:
        y_adj.append(1)
    elif v == 1:
        y_adj.append(-1)
    else:
        y_adj.append(0)
        
##accuracy
linear_accuracy = 100*f1_score(y_test, y_adj, average='weighted')
print(linear_accuracy)

##make test file
test_pred= test[train_cls]
pred_values = kmeans.predict(test_pred)
y_adj = []
y_adj  = [w-1 for w in pred_values]

id_v = test['id']  
df_final_linear = pd.DataFrame({'id':id_v,'sentiment_class':y_adj})    
df_final_linear.to_csv("../submissions/kmeans3.csv",index = False)

## Best model
## Linear regression  div = 0.33
## More models to try: XGBoost, Random Forest etc.




















