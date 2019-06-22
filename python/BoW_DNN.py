import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig("../build/plots/history_bow_knn.pdf")
    plt.clf()

seed = 42
dim = 100

X_train = np.genfromtxt("../build/preprocessed/bow_X_train.txt")
y_train = np.genfromtxt("../build/preprocessed/bow_y_train.txt", unpack=True)
X_test  = np.genfromtxt("../build/preprocessed/bow_X_test.txt")
y_test  = np.genfromtxt("../build/preprocessed/bow_y_test.txt", unpack=True)

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=dim))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

filepath = '../model/best_bow_nn.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, validation_split=0.3,
                    epochs=50, batch_size=8, callbacks=[checkpoint, TensorBoard(log_dir='../build/graph',
                                                                                histogram_freq=50, write_graph=True)])
print(history.history.keys())
plot_history(history)

#best_model = load_model('../model/best_bow_nn.hdf5')
#y_pred = best_model.predict(X_test, batch_size=64, verbose=1)
#y_pred_bool = np.argmax(y_pred, axis=1)
#print(classification_report(y_test, y_pred_bool))
#print(confusion_matrix(y_test, y_pred_bool, labels=[0, 1]))
#plt.imshow(confusion_matrix(y_test, y_pred_bool, labels=[0, 1]))
#plt.tight_layout()
#plt.colorbar()
#plt.savefig("../build/plots/cnfsn_mtx_bow_nn.pdf")
