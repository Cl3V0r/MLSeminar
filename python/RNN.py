import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, LeakyReLU, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig("../build/plots/history_lstm_loss.pdf")
    plt.clf()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("../build/plots/history_lstm_acc.pdf")
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
    print(confusion_matrix(Y_test, Y_cls, labels=[0, 1]))
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
    plt.savefig("../build/plots/hist_rnn_test.pdf")
    plt.clf()



df = pd.read_csv("../build/preprocessed/labeled_content_lem_stop.csv")
df = df.dropna()
#df = df.iloc[0:100]
X = df["content"]
y = df["label"]
print(np.count_nonzero(y==1),np.count_nonzero(y==0),len(y))
top_words = 5000
max_text_length = 500
seed=42

tokenizer = Tokenizer(num_words=top_words, filters=r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=max_text_length)
print('Shape of data tensor:', data.shape)

x_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.3, random_state=seed, shuffle=True) #,stratify=y)
print(np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 0), len(y_train))
print(np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0), len(y_test))

X_train, X_val, Y_train, y_val = train_test_split(x_train, y_train, test_size=0.3,
                                                 random_state=seed, shuffle=True)#, stratify=y)

filepath = '../model/best_rnn.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True)

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length,input_length=max_text_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(X_train, Y_train, validation_data=(X_val,y_val),
#                    epochs=100, batch_size=8, callbacks=[checkpoint,
#                     TensorBoard(log_dir='../build/graph', histogram_freq=50, write_graph=True)])
#plot_history(history)

best_model = load_model('../model/best_rnn.hdf5')
evaluate(X_test,y_test,X_train,Y_train,best_model)

y_pred = best_model.predict(X_test, batch_size=8, verbose=1)
y_pred_bool = best_model.predict_classes(X_test, batch_size=8, verbose=1)

plt.imshow(confusion_matrix(y_test, y_pred_bool,labels=[0, 1]))
plt.tight_layout()
plt.colorbar()
plt.xticks(range(2), ["fake", "real"])
plt.yticks(range(2), ["fake", "real"])
plt.savefig("../build/plots/cnfsn_mtx_rnn_test.pdf")
plt.clf()
#y_pred = best_model.predict(X_val, batch_size=8, verbose=1)
#y_pred_bool = best_model.predict_classes(X_val, batch_size=8, verbose=1)
#print(classification_report(y_val, y_pred_bool))
#print(confusion_matrix(y_val, y_pred_bool,labels=[0, 1]))
#plt.imshow(confusion_matrix(y_val, y_pred_bool,labels=[0, 1]))
#plt.tight_layout()
#plt.colorbar()
#plt.xticks(range(2), ["fake", "real"])
#plt.yticks(range(2), ["fake", "real"])
#plt.savefig("../build/plots/cnfsn_mtx_rnn_val.pdf")
#plt.clf()

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.show()
plt.savefig("../build/plots/bow/RNN_bow_roc.pdf")
plt.close()

X_false = []
for i in range(len(y_test)):
    if(y_test.iloc[i]!=y_pred_bool[i]):
       X_false.append(X.iloc[i])
    if(len(X_false)==3):
        break   

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

false_texts = list(map(sequence_to_text, X_test))
for f in false_texts:
    print("###############################################################")
    f=list(filter(None,f))
    print(' '.join(f))    