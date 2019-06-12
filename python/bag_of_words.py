import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])


seed=42
dim = 1000

df = pd.read_csv("../build/preprocessed/labeled_content_lem.csv")
df = df.dropna()
#df["label"] = df["label"].replace("fake", 0)
#df["label"] = df["label"].replace("real", 1)

print(df.keys())
print(df.head())
print(df.label.unique())

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

vectorizer = CountVectorizer(max_features=dim, ngram_range=(1, 3))

vectorizer.fit(X_train)
print(vectorizer.get_feature_names())
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)


model = Sequential()
model.add(Dense(units=11, activation='relu', input_dim=dim))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

filepath='../model/best_bow_nn.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, validation_split=0.3,
                    epochs=500, batch_size=8, callbacks=[checkpoint, TensorBoard(log_dir='../build/graph',
                                                                                 histogram_freq=0, write_graph=False)])
print(history.history.keys())
plot_history(history)
plt.show()
