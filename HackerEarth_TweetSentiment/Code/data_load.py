# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:22:00 2020

@author: a.sijaria
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

##Set the current directory to working folder
os.getcwd()
os.chdir("C:/Users/<curr_directory>/sentiment analysis/code")

##Loading dataset
train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")

##Looking for missing values
train.info()
test.info()


## nan values in lang and retweet_count
## imputing droping lang variables (only en as value)
train.drop('lang',axis=1,inplace= True)
test.drop('lang',axis=1,inplace= True)

## imputing 0 to retweet values (4 in train missing 1 in test missing)
train.retweet_count = train.retweet_count.fillna('0')
test.retweet_count = test.retweet_count.fillna('0')


## Variables assignement has been shifted to next variable
## in train value assignment has shifted, lanf applied to retweet_count
## and retweet_count assigned to original author

## fixing retweet counts
train['retweet_count'] = train.apply(lambda x: x['original_author'] if x['retweet_count'] == 'en' else x['retweet_count'],axis = 1 )
test['retweet_count'] = test.apply(lambda x: x['original_author'] if x['retweet_count'] == 'en' else x['retweet_count'],axis = 1 )

## removing gibberish values assigned
train['retweet_count'] = train.apply(lambda x: 0 if len(x['retweet_count'])>4  else x['retweet_count'],axis = 1)
test['retweet_count'] = test.apply(lambda x: 0 if len(x['retweet_count'])>4  else x['retweet_count'],axis = 1)

##changing retweet class to int
train.retweet_count = pd.to_numeric(train.retweet_count)
test.retweet_count = pd.to_numeric(test.retweet_count)

## Text proccessing on Tweets 

## changing everything to lower case
train['original_text'] = train.apply(lambda x: x['original_text'].lower(),axis=1)
test['original_text'] = test.apply(lambda x: x['original_text'].lower(),axis=1)

## removing punctuations
train['original_text'] = train.apply(lambda x: x['original_text'].translate(str.maketrans('','',string.punctuation)),axis=1)
test['original_text'] = test.apply(lambda x: x['original_text'].translate(str.maketrans('','',string.punctuation)),axis=1)

## removing digits
train['original_text'] = train.apply(lambda x: x['original_text'].translate(str.maketrans('', '', digits)) ,axis=1)
test['original_text'] = test.apply(lambda x: x['original_text'].translate(str.maketrans('', '', digits) ) ,axis=1)

## Most of the text cleaning has been done, post observing frequency of the data set
### remove other characted like '...' and 'xxxxx' from words
train['original_text'] = train.apply(lambda x: x['original_text'].translate(str.maketrans('', '', '.')) ,axis=1)
train['original_text'] = train.apply(lambda x: x['original_text'].translate(str.maketrans('', '', 'xxxxx')) ,axis=1)
train['original_text'] = train.apply(lambda x: x['original_text'].translate(str.maketrans('', '', 'xxx')) ,axis=1)

test['original_text'] = test.apply(lambda x: x['original_text'].translate(str.maketrans('', '', '.')) ,axis=1)
test['original_text'] = test.apply(lambda x: x['original_text'].translate(str.maketrans('', '', 'xxxxx')) ,axis=1)
test['original_text'] = test.apply(lambda x: x['original_text'].translate(str.maketrans('', '', 'xxx')) ,axis=1)

##tokenize words
train['new_text'] = train.apply(lambda x: word_tokenize(x['original_text']),axis=1)
test['new_text'] = test.apply(lambda x: word_tokenize(x['original_text']),axis=1)

##lemmatize verbs and nouns
## in final model only nouns have been lemmatize, verbs used to indicate polarity
def lemmat_words(words,pos):
    lemmated = [lemmatizer.lemmatize(word,pos) for word in words]
    return lemmated

lemmatizer = WordNetLemmatizer()
train['clean_text_lem'] = train.apply(lambda x: lemmat_words(x['new_text'],pos = wordnet.NOUN),axis=1)
test['clean_text_lem'] = test.apply(lambda x: lemmat_words(x['new_text'],pos = wordnet.NOUN),axis=1)

## removed from the final model
## train['clean_text_lem'] = train.apply(lambda x: lemmat_words(x['clean_text_lem'],pos = wordnet.VERB),axis=1)
## test['clean_text_lem'] = test.apply(lambda x: lemmat_words(x['clean_text_lem'],pos = wordnet.VERB),axis=1)

## not used in final model
## stemming words

porter = PorterStemmer()
def stem_words(words):
    stemmed = [porter.stem(word) for word in words]
    return stemmed

train['clean_text_stem'] = train.apply(lambda x: stem_words(x['new_text']),axis=1)
test['clean_text_stem'] = test.apply(lambda x: stem_words(x['new_text']),axis=1)

##remove stopwords

## stop words from nltk module
stop_words = list(set(stopwords.words('english')))

## stop words based on frequency table
stop_words2 = ['’','httpswww','httpswww','…','https','like','may','get','would','im'
               ,'x','youre','u','xx','xxx','mam','”','•','put','“','http','‘','…happy','happy']

##stop words not contributing to the tone (theme words)
stop_words3 = ['day','mothers','mothersday','mother',
               'mum','mums','women','special','motheringsunday','mummy','mom'
               ,'mothering','motherhood']


def rm_stopwords(text_words):
    stopped1 = [w for w in text_words if not w in stop_words]
    stopped2 = [w for w in stopped1 if not w in stop_words2]
    stopped3 = [w for w in stopped2 if not w in stop_words3]
    return stopped3

train['clean_text'] = train.apply(lambda x: rm_stopwords(x['clean_text_lem']),axis=1)
test['clean_text'] = test.apply(lambda x: rm_stopwords(x['clean_text_lem']),axis=1)

##removing big non meaningful words words with len>=15 not adding any meaning
def big_words(text_words):
    keep = [w for w in text_words if not len(w)>=15]
    #print(keep)
    return keep

train['clean_text'] = train.apply(lambda x: big_words(x['clean_text']),axis=1)
test['clean_text'] = test.apply(lambda x: big_words(x['clean_text']),axis=1)

## check
#a = train.clean_text.sum()
#a = word_tokenize(a)
#keep = [w for w in a if len(w)==15]


## seeing the count of keywords in tweets
train['cnt_words'] = train.apply(lambda x: len(x.clean_text),axis = 1)
test['cnt_words'] = test.apply(lambda x: len(x.clean_text),axis = 1)

## len of message
## count of words in original tweets
train['ttl_wrds'] = train.apply(lambda x: len(x.new_text),axis = 1)
test['ttl_wrds'] = test.apply(lambda x: len(x.new_text),axis = 1)

##orginal author
##changind everything to lower case
train['original_author'] = train.apply(lambda x: x['original_author'].lower(),axis=1)
test['original_author'] = test.apply(lambda x: x['original_author'].lower(),axis=1)


##creating only verbs/JJ/etc column - which are polarizing words
def pos_tag(words):
    text = ' '.join([w for w in words])
    blob= TextBlob(text)
    wrds = blob.sentiment_assessments.assessments
    return ([y for x in (w[0] for w in wrds) for y in x])

train['clean_text_new'] = train.apply(lambda x: pos_tag(x['clean_text']),axis=1)
test['clean_text_new'] = test.apply(lambda x: pos_tag(x['clean_text']),axis=1)


##count of polazing words
train['cnt_words_new'] = train.apply(lambda x: len(x.clean_text_new),axis = 1)
test['cnt_words_new'] = test.apply(lambda x: len(x.clean_text_new),axis = 1)

##adding polarity and subjectivty of text
def polarity(words):
    text = ' '.join([w for w in words])
    blob = TextBlob(text)
    return(blob.polarity)

train['clean_text_polarity'] = train.apply(lambda x: polarity(x['clean_text']),axis=1)
train['original_text_polarity'] = train.apply(lambda x: polarity(x['original_text']),axis=1)
train['clean_text_lem_polarity'] = train.apply(lambda x: polarity(x['clean_text_lem']),axis=1)
train['clean_text_stem_polarity'] = train.apply(lambda x: polarity(x['clean_text_stem']),axis=1)
test['clean_text_polarity'] = test.apply(lambda x: polarity(x['clean_text']),axis=1)
test['original_text_polarity'] = test.apply(lambda x: polarity(x['original_text']),axis=1)
test['clean_text_lem_polarity'] = test.apply(lambda x: polarity(x['clean_text_lem']),axis=1)
test['clean_text_stem_polarity'] = test.apply(lambda x: polarity(x['clean_text_stem']),axis=1)

def subj(words):
    text = ' '.join([w for w in words])
    blob = TextBlob(text)
    return(blob.subjectivity)

train['clean_text_subj'] = train.apply(lambda x: subj(x['clean_text']),axis=1)
train['original_text_subj'] = train.apply(lambda x: subj(x['original_text']),axis=1)
train['clean_text_lem_subj'] = train.apply(lambda x: subj(x['clean_text_lem']),axis=1)
train['clean_text_stem_subj'] = train.apply(lambda x: subj(x['clean_text_stem']),axis=1)
test['clean_text_subj'] = test.apply(lambda x: subj(x['clean_text']),axis=1)
test['original_text_subj'] = test.apply(lambda x: subj(x['original_text']),axis=1)
test['clean_text_lem_subj'] = test.apply(lambda x: subj(x['clean_text_lem']),axis=1)
test['clean_text_stem_subj'] = test.apply(lambda x: subj(x['clean_text_stem']),axis=1)


train.columns

##creating clean dataset
train.to_csv("../dataset/train_cleaned.csv")
test.to_csv("../dataset/test_cleaned.csv")






