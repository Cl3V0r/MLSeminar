import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import numpy as np
from PIL import Image
from PIL import ImageFilter

def plotWordcloud(s,t):
    if(t!=""):
       mask = np.array(Image.open('../data/pictures/'+t))
    else:
        mask=None

    contents = Path('../build/preprocessed/'+s+".csv").read_text()
    wordcloud = WordCloud(background_color='black',
                      width=1920,
                      height=1080,
                      mask=mask
                      ).generate(contents)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.savefig("../build/plots/"+s+"_wordcloud.pdf")
    plt.clf()


plotWordcloud("fake_news_titles_stem","trump_silhouette.png")
plotWordcloud("fake_news_titles_lem", "trump_silhouette.png")
plotWordcloud("real_news_titles_lem","statue_of_liberty.png")

X_train = np.genfromtxt("../build/preprocessed/bow_X_train.txt")
size = len(X_train)
X_train = np.sum(X_train, axis=0)/size
y_train = np.genfromtxt("../build/preprocessed/bow_y_train.txt", unpack=True)
X_test = np.genfromtxt("../build/preprocessed/bow_X_test.txt")
size = len(X_test)
X_test = np.sum(X_test, axis=0)/size
y_test = np.genfromtxt("../build/preprocessed/bow_y_test.txt", unpack=True)
keys=[]
dim=1000
with open("../build/preprocessed/bow_feature_names.txt", "r") as file:
    for line in file:
        current = line[:-1]
        keys.append(current)
weights=np.arange(0,dim)        
plt.bar(weights,X_train,alpha=0.8,label="train", color="r")
plt.bar(weights,X_test, alpha=0.4, label="test", color="b")
plt.xticks(range(len(keys)), keys, rotation='vertical', size='small')
plt.legend()
plt.tight_layout()
plt.savefig("../build/plots/comparisonX_train_test.pdf")
plt.clf()
plt.hist(y_train,density=True,color="r",alpha=0.4,label="train")
plt.hist(y_test, density=True, color="b", alpha=0.4,label="test")
plt.xticks(range(2), ["fake","real"])
plt.legend()
plt.tight_layout()
plt.savefig("../build/plots/comparisonY_train_test.pdf")
plt.clf()

X_train = np.genfromtxt("../build/preprocessed/bow_X_train.txt")
X_fake=[]
X_real=[]
for i in range(len(X_train)):
   if(y_train[i]==0):
       X_fake.append(X_train[i,:])
   else:
       X_real.append(X_train[i,:])     
size = len(X_fake)
X_fake=np.sum(X_fake,axis=0)/size
size = len(X_real)
X_real = np.sum(X_real, axis=0)/size
fig = plt.figure(figsize=(24, 20))
plt.bar(weights, X_fake, alpha=0.8, label="fake", color="r")
plt.bar(weights, X_real, alpha=0.4, label="real", color="b")
plt.xticks(range(len(keys)), keys, rotation=30, size=1)
plt.legend()
plt.tight_layout()
plt.savefig("../build/plots/comparisonX_fake_real.pdf")
plt.clf()
