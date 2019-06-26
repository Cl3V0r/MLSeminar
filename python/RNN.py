import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

df = pd.read_csv("../build/preprocessed/labeled_content_lem_stop.csv")
df = df.dropna()
#df=df[:1000]
#get newstext with label: real or fake
X = df["content"]
y = df["label"]
