# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:05 2022

@author: hardh
"""

import numpy as np
import pandas as pd
import time

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

#tf.random.set_seed(15)

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

#Training

#embed_vec_len = 64
total_log_keys = 29

x_train = x_train.reshape((len(valid_seq), max_seq_len, 1))
x_test = x_test.reshape((len(train_seq), max_seq_len, 1))
#x_train = x_train.reshape((len(train_seq), max_seq_len, 1))
#x_test = x_test.reshape((len(valid_seq), max_seq_len, 1))
x_train = x_train[:2000,:,:]
y_train = y_train[:2000]

model = Sequential()
#model.add(Embedding(total_log_keys, embed_vec_len, input_length=max_seq_len))
model.add(LSTM(50, input_shape=(max_seq_len, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
#sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, verbose=1)  

#Plot Model Accuracy

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

"""
def plot_graphs(history, string):
    figure(figsize=(8, 6), dpi=80)
    plt.plot(np.arange(1,21), history.history[string], marker="x", linewidth=3, \
             markersize=10)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel(string, fontsize=20)
    plt.xlim(1,20)
    plt.xticks(np.arange(1,21, step=1), size=15)
    plt.yticks(size=15)
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
"""
def plot_graphs(history, string):
    figure(figsize=(8, 6), dpi=80)
    plt.plot(np.arange(1,11), history.history[string], marker="x", linewidth=3, \
             markersize=10)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel(string, fontsize=20)
    plt.xlim(1,10)
    plt.xticks(np.arange(1,11, step=1), size=15)
    plt.yticks(size=15)
    plt.grid()
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

# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_no_quant.tflite"
convert_to_tflite_noquant(model, model_tflite_filename)


interpreter_noquant = tf.lite.Interpreter("model_no_quant.tflite")
interpreter_noquant.allocate_tensors()

y_pred_noquant = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_noquant.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_noquant = np.array([1 if x > 0.5 else 0 for x in y_pred_noquant])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred_noquant == y_test) / len(y_pred_noquant), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred_noquant))

# pred hybrid quant
def convert_to_tflite_hybridquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_hybrid_quant.tflite"
convert_to_tflite_hybridquant(model, model_tflite_filename)


interpreter_hybridquant = tf.lite.Interpreter("model_hybrid_quant.tflite")
interpreter_hybridquant.allocate_tensors()

y_pred_hybrid = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred_hybrid.append(pred[0][0])

print("---Pred time (hybrid):  %s seconds ---" % (time.time() - start_time))
    
y_pred_hybrid = np.array([1 if x > 0.5 else 0 for x in y_pred_hybrid])
print("TPU accuracy (hybrid): ", 100 * np.sum(y_pred_hybrid == y_test) / len(y_pred_hybrid), "%")
print("F1 score (hybrid): ", f1_score(y_test, y_pred_hybrid))

#pred int quant
def representative_dataset():
  # To ensure full coverage of possible inputs, we use the whole train set
  for i in range(x_train.shape[0]):
    input_data = x_train[i, :, :].reshape((1, max_seq_len, 1))
    input_data = tf.dtypes.cast(input_data, tf.float32)
    yield [input_data]

batch_size = 1
model.input.set_shape((batch_size,) + model.input.shape[1:])

def convert_to_tflite_int(model, filename):
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity
    converter.target_spec.supported_types = [tf.int8]
    """
    # These set the input and output tensors to int8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    """
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_int_quant.tflite"
convert_to_tflite_int(model, model_tflite_filename)

#Pred hybrid
interpreter_int = tf.lite.Interpreter("model_int_quant.tflite")
interpreter_int.allocate_tensors()

y_pred_int = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, 1)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_int.append(pred[0][0])

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_int = np.array([1 if x > 0.5 else 0 for x in y_pred_int])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == y_test) / len(y_pred_int), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred_int))

"""
#Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))
"""
