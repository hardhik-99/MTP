# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:05 2022

@author: hardh
"""

import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
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

#x_train = np.array(pad_sequences(train_seq, maxlen=max_seq_len, padding='pre'))
#x_test = np.array(pad_sequences(valid_seq, maxlen=max_seq_len, padding='pre'))

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
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

#Training

embed_vec_len = 64
total_log_keys = 29

model = Sequential()
model.add(Embedding(total_log_keys, embed_vec_len, input_length=max_seq_len))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

#adam = Adam(lr=0.1)
sgd = SGD(learning_rate=0.05)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=20, verbose=1)  

#Plot Model Accuracy

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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

#Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))
"""
x_train_seq = pad_sequences(train_seq, maxlen=max_seq_len, padding='pre')
dataset = tf.data.Dataset.from_tensor_slices(x_train_seq).batch(1)
#TFlite
def representative_data_gen():
  # To ensure full coverage of possible inputs, we use the whole train set
  for input_data in dataset:
    input_data = tf.dtypes.cast(input_data, tf.float32)
    yield [input_data]
"""
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    """
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # These set the input and output tensors to int8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    """

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_no_quant.tflite"
convert_to_tflite_noquant(model, model_tflite_filename)

def convert_to_tflite_hybrid(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    """
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # These set the input and output tensors to int8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    """
    # Set the optimization mode 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model_hybrid_quant.tflite"
convert_to_tflite_hybrid(model, model_tflite_filename)


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

