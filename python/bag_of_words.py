import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

seed=42
dim = 1000
#get lemmatized dataset (needs preprocessing.py)
df = pd.read_csv("../build/preprocessed/labeled_content_lem_stop.csv")
df = df.dropna()
#df=df[:1000]
#get newstext with label: real or fake
X = df["content"]
y = df["label"]

#split dataset into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, shuffle=True) #,stratify=y)
#actual bag of words model, with most common words (n=dim) of trainigsdataset
vectorizer = CountVectorizer(max_features=dim, ngram_range=(1,1))
vectorizer.fit(X_train)

with open("../build/preprocessed/bow_feature_names.txt","w") as file:
    for vec in vectorizer.get_feature_names():
        file.write(vec+"\n")

X_train=vectorizer.transform(X_train).toarray()
X_test=vectorizer.transform(X_test).toarray()

np.savetxt("../build/preprocessed/bow_X_train.txt",X_train)
np.savetxt("../build/preprocessed/bow_y_train.txt",y_train)
np.savetxt("../build/preprocessed/bow_X_test.txt",X_test)
np.savetxt("../build/preprocessed/bow_y_test.txt",y_test)

#visualize data. use tSNE for a 3 dim representation. try to separate 3dim data with kmeans 
X_embedded = TSNE(n_components=3).fit_transform(X_train[:4000])
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_embedded)
#print(kmeans.labels_)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, label in enumerate(y_train[:4000]):
        x, y, z = X_embedded[i, :]
        ax.text(X_embedded[:, 0][i], X_embedded[:, 1][i], X_embedded[:, 2][i], label)
ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=kmeans.labels_, cmap='rainbow')
ax.scatter3D(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], color='black')

ax.set_xlim(-70, 70)
ax.set_ylim(-70, 70)
ax.set_zlim(-70, 70)
plt.savefig("../build/plots/tSNE_bow_knn.pdf")
plt.clf()

# display 2D scatter plot
#for i, label in enumerate(y_train[:500]):
#        x, y = X_embedded[i, :]
#        plt.annotate(label, xy=(x, y), xytext=(
#                    0, 0), textcoords='offset points')
#
#plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
#            c=kmeans.labels_, cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:, 0],
#            kmeans.cluster_centers_[:, 1], color='black')
