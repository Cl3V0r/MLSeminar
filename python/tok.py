#!usr/bin/env python
#coding:utf8
#from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.cistem import Cistem
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

nltk.download('stopwords')
tknzr= TweetTokenizer()
stemmer = Cistem()
file_in = open("../data/postillon.txt", "r")
file_out = open("../data/postillon_stem.txt", "w")
for line in file_in:
    tokenized = tknzr.tokenize(line)
    for word in tokenized:
        if word in stopwords.words('german'):
            tokenized.remove(word)
        word = stemmer.stem(word)
    token_text = " ".join(tokenized)   
    file_out.write(token_text+'\n')      
file_in.close()
file_out.close()

data = open("../data/postillon_stem.txt", "r")
vectorizer = CountVectorizer(max_features=100, ngram_range=(2, 2))
X = vectorizer.fit_transform(data).toarray()
print(vectorizer.get_feature_names())
print(X)

contents = Path("../data/postillon_stem.txt").read_text()
wordcloud = WordCloud(background_color='white',
                      width=1200,
                      height=1000
                      ).generate(contents)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
