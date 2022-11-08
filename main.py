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
model.add(LSTM(30))
#model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=2, verbose=1)  
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

#TFlite
def convert_to_tflite(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "model.tflite"
convert_to_tflite(model, model_tflite_filename)

#Create quantized version
batch_size = 1
model.input.set_shape((batch_size,) + model.input.shape[1:])
model.summary()

# Our representative dataset is the same as the training dataset,
# but the batch size must now be 1
dataset_repr = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=max_seq_len,
    batch_size=batch_size,
)

def representative_data_gen():
  # To ensure full coverage of possible inputs, we use the whole train set
  for input_data, _ in dataset_repr.take(int(len(x_train))):
    input_data = tf.cast(input_data, dtype=tf.float32)
    yield [input_data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to int8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

with open('model_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)

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

