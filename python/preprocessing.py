import pandas as pd 
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
tknzr = TweetTokenizer()
stemmer = PorterStemmer()

news = pd.read_csv('../data/mixed_news/news_dataset.csv')
print(news.keys())
print(news.head())
print(news[news['label']=='fake'].title)