#!usr/bin/env python
#coding:utf8

import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')
tknzr = TweetTokenizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def stemming(s):
    tokenized = tknzr.tokenize(s)
    l = []
    for word in tokenized:
        if word in stopwords.words('english'):
            tokenized.remove(word)
        l.append(stemmer.stem(word))
    token_text = " ".join(l)
    return token_text


def lemmatizing(s, stop):
    if(type(s) != str):
        print(type(s), " ", s)
        return "nan"
    tokenized = tknzr.tokenize(s)
    l = []
    for word in tokenized:
        if(stop == True):
            if word in stopwords.words('english'):
                tokenized.remove(word)
        word = lemmatizer.lemmatize(word)
    token_text = " ".join(tokenized)
    return token_text


#getting to know the dataset
news = pd.read_csv('../data/mixed_news/news_dataset.csv')

print(news.keys())
print(news.head())
print(news.label.unique())

#extracting titles for wordcloud visualization
titles = news.title.dropna()
fake_titles = titles[news['label'] == 'fake']
real_titles = titles[news['label'] == 'real']
stem_data = open('../build/preprocessed/fake_news_titles_stem.csv', "w")
lem_data = open('../build/preprocessed/fake_news_titles_lem.csv', "w")
for t in fake_titles:
    stem_data.write(stemming(t)+"\n")
    lem_data.write(lemmatizing(t, True)+"\n")
stem_data.close()
lem_data.close()
lem_data = open('../build/preprocessed/real_news_titles_lem.csv', "w")
for t in real_titles:
    lem_data.write(lemmatizing(t, True)+"\n")
lem_data.close

#lemmatize articles
#data = open("../build/preprocessed/labeled_content_lem.csv","w")
#for row in news.itertuples():
#   data.write(lemmatizing(getattr(row, "content"))+" "+getattr(row, "label")+"\n")
#data.close()
news['content'] = news['content'].dropna()
news["label"] = news["label"].replace("fake", 0)
news["label"] = news["label"].replace("real", 1)
news['content'] = news['content'].apply(lambda x: lemmatizing(x, False))
news.to_csv(
    path_or_buf="../build/preprocessed/labeled_content_lem.csv", index=False)
news['content'] = news['content'].apply(lambda x: lemmatizing(x, True))
news.to_csv(
    path_or_buf="../build/preprocessed/labeled_content_lem_stop.csv", index=False)
