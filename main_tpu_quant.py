# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:05 2022

@author: hardh
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm
import time

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

x_test = np.array(pad_sequences(train_seq, maxlen=max_seq_len, padding='pre'))
x_train = np.array(pad_sequences(valid_seq, maxlen=max_seq_len, padding='pre'))
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
"""
y_train = np.array(train_labels)
y_test = np.array(valid_labels)
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)
"""
y_test = np.array(train_labels)
y_train = np.array(valid_labels)
y_train = np.asarray(y_train).astype(np.int8)
y_test = np.asarray(y_test).astype(np.int8)

#Load TFlite model
import tflite_runtime.interpreter as tflite

tflite_filename = 'model_no_quant.tflite'

def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(tflite_filename)

#Run the model on TPU
def tpu_tflite_predict(interpreter, data):
    #input_data = data.reshape(1, max_seq_len).astype(np.float32)
    input_data = data.reshape(1, max_seq_len).astype(np.int8)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

y_pred = []

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    pred = tpu_tflite_predict(interpreter, x_test[i])
    y_pred.append(pred[0][0])

print("---Pred time (no quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (no quant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (no quant): ", f1_score(y_test, y_pred))

tflite_filename = 'model_hybrid_quant.tflite'

def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(tflite_filename)

#Run the model on TPU
def tpu_tflite_predict(interpreter, data):
    #input_data = data.reshape(1, max_seq_len).astype(np.float32)
    input_data = data.reshape(1, max_seq_len).astype(np.int8)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

y_pred = []

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    pred = tpu_tflite_predict(interpreter, x_test[i])
    y_pred.append(pred[0][0])

print("---Pred time (hybrid quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (hybrid quant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (hybrid quant): ", f1_score(y_test, y_pred))