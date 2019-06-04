#!usr/bin/env python
#coding:utf8
#from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.cistem import Cistem

stemmer = Cistem()
s1 = "Speicherbehältern"
stemmer.stem(s1)
s2 = "Grenzpostens"
stemmer.stem(s2)
s3 = "Ausgefeiltere"
stemmer.stem(s3)
print(s1,s2,s3)
stemmer = Cistem(True)
print(stemmer.stem(s1),
stemmer.stem(s2),
stemmer.stem(s3))