import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig("../build/plots/history_bow_dnn.pdf")
    plt.clf()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("../build/plots/history_bow_dnn_acc.pdf")
    plt.clf()

def evaluate(X_test, Y_test, X_train, Y_train, model):
    ##Evaluate loss and metrics and predict & classes
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    Y_pred = model.predict(X_test, batch_size=1)
    Y_cls = model.predict_classes(X_test, batch_size=1)
    print('Test Loss:', loss)
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_cls))
    print("Precision: %.2f" % precision_score(Y_test, Y_cls, average='weighted'))
    print("Recall: %.2f" % recall_score(Y_test, Y_cls, average='weighted'))
    print('Classification Report:\n', classification_report(Y_test, Y_cls) )
    #print(confusion_matrix(y_val, y_pred, labels=[0, 1]))
    ## Plot 0 probability including overtraining test
    plt.figure(figsize=(8, 8))
    label = 1
    #Test prediction
    plt.hist(Y_pred[Y_test == label], alpha=0.5,
             color='red', range=[0, 1], bins=10)
    plt.hist(Y_pred[Y_test != label], alpha=0.5,
             color='blue', range=[0, 1], bins=10)
    #Train prediction
    Y_train_pred = model.predict(X_train)
    plt.hist(Y_train_pred[Y_train == label], alpha=0.5, color='red', range=[
             0, 1], bins=10, histtype='step', linewidth=2)
    plt.hist(Y_train_pred[Y_train != label], alpha=0.5, color='blue', range=[
             0, 1], bins=10, histtype='step', linewidth=2)
    plt.legend(['train == 1', 'train == 0', 'test == 1',
                'test == 0'], loc='upper right')
    plt.xlabel('Probability of being real news')
    plt.ylabel('Number of entries')
    plt.savefig("../build/plots/hist_bow_dnn_test.pdf")    

seed = 42
rate=0.3
dim = 1000

X_train = np.genfromtxt("../build/preprocessed/bow_X_train.txt")
y_train = np.genfromtxt("../build/preprocessed/bow_y_train.txt", unpack=True)
X_test  = np.genfromtxt("../build/preprocessed/bow_X_test.txt")
y_test  = np.genfromtxt("../build/preprocessed/bow_y_test.txt", unpack=True)

LR = LeakyReLU()
LR.__name__ = 'relu'

model = Sequential()
model.add(Dense(units=dim*10, activation=LR, input_dim=dim))
#model.add(Dense(units=1000, activation='relu'))
model.add(Dropout(rate=rate,seed=seed))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

filepath = '../model/best_bow_nn.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.3,
                    epochs=1000,batch_size=8, callbacks=[checkpoint, TensorBoard(log_dir='../build/graph',
                                                                                histogram_freq=50, write_graph=True)])
print(history.history.keys())
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(path_or_buf = "../build/history_bow_dnn.csv",index=False)
plot_history(history)

best_model = load_model('../model/best_bow_nn.hdf5')
evaluate(X_test,y_test,X_train,y_train,best_model)
y_pred = best_model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.round(y_pred)
print(y_pred,y_pred_bool)
print(classification_report(y_test, y_pred_bool))
print(confusion_matrix(y_test, y_pred_bool,
                       labels=[0,1]))
plt.imshow(confusion_matrix(y_test, y_pred_bool,
                            labels=[0,1]))
plt.tight_layout()
plt.colorbar()
plt.xticks(range(2), ["fake", "real"])
plt.yticks(range(2), ["fake", "real"])
plt.savefig("../build/plots/cnfsn_mtx_bow_nn.pdf")
