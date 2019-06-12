import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

news = pd.read_csv("../build/preprocessed/labeled_content_lem.csv")
