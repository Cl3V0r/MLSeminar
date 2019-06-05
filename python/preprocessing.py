import pandas as pd 
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
tknzr = TweetTokenizer()
stemmer = PorterStemmer()

news = pd.read_csv('../data/mixed_news/news_dataset.csv')
print(news.keys())
print(news.head())

fake_titles = news[news['label'] == 'fake'].title.dropna()
fake_titles.to_csv('../build/preprocessed/fake_news_titles.csv',index=False,sep=" ",header=False)