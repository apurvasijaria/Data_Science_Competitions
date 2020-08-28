# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 18:42:29 2020

@author: apurva.sijaria
"""

import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import digits 
import string
from textblob import TextBlob


##get working directory 
os.getcwd()
# os.chdir('C:\\Users\\apurv\\OneDrive\\Documents\\Projects\\2020\\Data_Science_Competitions\\Kaggle_RealOrNot\\Code')

##loading dataset
train = pd.read_csv('..\\data\\train.csv')
test = pd.read_csv('..\\data\\test.csv')

##Looking for missing values
train.info()
test.info()
#location and keyword have missing values


train.describe()




