#!usr/bin/env python
#coding:utf8

import pandas as pd 
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
tknzr = TweetTokenizer()
stemmer = PorterStemmer()

#getting to know the dataset
news = pd.read_csv('../data/mixed_news/news_dataset.csv')
print(news.keys())
print(news.head())
print(news.label.unique())

#extracting titles for wordcloud visualization
titles = news.title.dropna()

for t in titles:
    tokenized = tknzr.tokenize(t)
    for word in tokenized:
        if word in stopwords.words('english'):
            tokenized.remove(word)
        word = stemmer.stem(word)
    token_text = " ".join(tokenized)
    t = token_text       

fake_titles = titles[news['label'] == 'fake']
fake_titles.to_csv('../build/preprocessed/fake_news_titles.csv',index=False,sep=" ",header=False)

real_titles = titles[news['label'] == 'real']
real_titles.to_csv('../build/preprocessed/real_news_titles.csv',
                   index=False, sep=" ", header=False)
