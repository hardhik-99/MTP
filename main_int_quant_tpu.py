# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:05 2022

@author: hardh
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adagrad

#Load TFlite model
import tflite_runtime.interpreter as tflite

#tf.random.set_seed(5)
#np.random.seed(11)

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

#x_train = np.array(pad_sequences(train_seq, maxlen=max_seq_len, padding='pre'))
#x_test = np.array(pad_sequences(valid_seq, maxlen=max_seq_len, padding='pre'))

x_test = np.array(pad_sequences(train_seq, maxlen=max_seq_len, padding='pre'))
x_train = np.array(pad_sequences(valid_seq, maxlen=max_seq_len, padding='pre'))
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

"""
y_train = np.array(train_labels)
y_test = np.array(valid_labels)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

"""
y_test = np.array(train_labels)
y_train = np.array(valid_labels)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


#embed_vec_len = 64
total_log_keys = 29

x_train = x_train.reshape((len(valid_seq), max_seq_len, 1))
x_test = x_test.reshape((len(train_seq), max_seq_len, 1))
#x_train = x_train.reshape((len(train_seq), max_seq_len, 1))
#x_test = x_test.reshape((len(valid_seq), max_seq_len, 1))
x_train = x_train[:2000,:,:]
y_train = y_train[:2000]



def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_no_quant.tflite"
interpreter_noquant = load_tflite_model(model_tflite_filename)
interpreter_noquant.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred))

# pred hybrid quant

model_tflite_filename = "model_hybrid_quant.tflite"
interpreter_hybridquant = load_tflite_model(model_tflite_filename)
interpreter_hybridquant.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred))

# pred int quant

model_tflite_filename = "model_int_quant.tflite"
interpreter_int = load_tflite_model(model_tflite_filename)
interpreter_int.allocate_tensors()

y_pred = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_int, x_test_sample)
    y_pred.append(pred[0][0])

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred == y_test) / len(y_pred), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred))

"""
#Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))
"""
