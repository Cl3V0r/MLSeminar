import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import TensorBoard

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])


seed=42
dim = 100

df = pd.read_csv("../build/preprocessed/labeled_content_lem.csv")
df = df.dropna()
#df = df[:1000]
print(df.keys())
print(df.head())
print(df.label.unique())

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, shuffle=True ,stratify=y)

vectorizer = CountVectorizer(max_features=dim, ngram_range=(1, 3))

vectorizer.fit(X_train)
print(vectorizer.get_feature_names())
X_train=vectorizer.transform(X_train).toarray()
X_test=vectorizer.transform(X_test).toarray()
print(y_train)
X_embedded = TSNE(n_components=3).fit_transform(X_train[:4000])
kmeans = KMeans(n_clusters=12)
kmeans.fit(X_embedded)
print(kmeans.labels_)
fig = plt.figure()
ax = Axes3D(fig)
for i, label in enumerate(y_train[:4000]):
        x, y, z = X_embedded[i, :]
        ax.text(X_embedded[:, 0][i], X_embedded[:, 1][i], X_embedded[:, 2][i], label)
ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
            c=kmeans.labels_, cmap='rainbow')
ax.scatter3D(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], color='black')
# display scatter plot
#for i, label in enumerate(y_train[:500]):
#        x, y = X_embedded[i, :]
#        plt.annotate(label, xy=(x, y), xytext=(
#                    0, 0), textcoords='offset points')
#
#plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
#            c=kmeans.labels_, cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:, 0],
#            kmeans.cluster_centers_[:, 1], color='black')

plt.savefig("../build/plots/tSNE_bow_knn.pdf")
plt.clf()

model = Sequential()
model.add(Dense(units=101, activation='relu', input_dim=dim))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

filepath='../model/best_bow_nn.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, validation_split=0.3,
                    epochs=100, batch_size=8, callbacks=[checkpoint, TensorBoard(log_dir='../build/graph',
                                                                                 histogram_freq=0, write_graph=False)])
print(history.history.keys())
plot_history(history)
plt.savefig("../build/plots/history_bow_knn.pdf")

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
