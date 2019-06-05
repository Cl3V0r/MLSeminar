#!usr/bin/env python
#coding:utf8
from nltk.tokenize import TweetTokenizer
from nltk.stem.cistem import Cistem
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

nltk.download('stopwords')
tknzr= TweetTokenizer()
stemmer = Cistem(True)
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
vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 3))
X = vectorizer.fit_transform(data).toarray()
print(vectorizer.get_feature_names())
print(X)

contents = Path("../data/postillon_stem.txt").read_text()
wordcloud = WordCloud(background_color='white',
                      width=1920,
                      height=1080
                      ).generate(contents)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

X_embedded = TSNE(n_components=2).fit_transform(X)

kmeans = KMeans(n_clusters=8)
kmeans.fit(X_embedded)
print(kmeans.labels_)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], color='black')
plt.show()


