# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:05 2022

@author: hardh
"""

import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

#Preprocessing

train = pd.read_csv('train.csv')
train = train.sample(frac=1).reset_index(drop=True).values

valid = pd.read_csv('valid.csv')
valid = valid.sample(frac=1).reset_index(drop=True).values

test =  pd.read_csv('test.csv')
test = test.sample(frac=1).reset_index(drop=True).values

train_seq, train_labels = train[:, 0], train[:, 1]
valid_seq, valid_labels = valid[:, 0], valid[:, 1]
test_seq, test_labels = test[:, 0], test[:, 1]

for i in range(len(train_seq)):
    train_seq[i] = [x for x in train_seq[i].split(' ')]
    
for i in range(len(valid_seq)):
    valid_seq[i] = [x for x in valid_seq[i].split(' ')]
    
for i in range(len(test_seq)):
    test_seq[i] = [x for x in test_seq[i].split(' ')]

max_seq_len = max([len(x) for x in train_seq])

x_train = np.array(pad_sequences(train_seq, maxlen=max_seq_len, padding='pre'))
x_test = np.array(pad_sequences(valid_seq, maxlen=max_seq_len, padding='pre'))
#x_test = np.array(pad_sequences(test_seq, maxlen=max_seq_len, padding='pre'))
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
#x_test = np.asarray(x_test)

y_train = np.array(train_labels)
y_test = np.array(valid_labels)
#y_test = np.array(test_labels)
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)
#y_test = np.asarray(y_test).astype(np.int32)

#Training

embed_vec_len = 64
total_log_keys = 29

model = Sequential()
model.add(Embedding(total_log_keys, embed_vec_len, input_length=max_seq_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, verbose=1)  
#validation_data=(x_valid, y_valid), verbose=2)

#Plot Model Accuracy

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))


#Save Model

save_path = r'D:\IIT KGP Study\EE 4yr\9th semester\MTP\MTP_code\save_model'

tf.saved_model.save(model, save_path)

#Convert TFlite

converter = tf.lite.TFLiteConverter.from_saved_model(save_path) 
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

"""
#Load TFlite model
import tflite_runtime.interpreter as tflite

tflite_filename = 'model.tflite'

def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(tflite_filename)

#Run the model on TPU
def tflite_predict(interpreter, data):
    return
"""

