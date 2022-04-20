# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:17:29 2022

@author: HP
"""

#   Project 1. 

#   Perform classification on breast cancer dataset.

#   Carry output the proper data preprocessing steps.

#   Create a feedforward neural network for classification.

#   Try to achieve a good level of accuracy (>80%)


import sklearn.datasets as skdatasets
import pandas as pd
import numpy as np
import datetime
import os 
from sklearn import preprocessing

#   READ CANCER DATASET
bc_data = pd.read_csv(r"C:\Users\HP\Desktop\Deep Learning with Python\4th week\Dataset Project\Breast Cancer Wisconsin (Diagnostic) Data Set\data.csv")
print(bc_data)
print("Breast Cancer (dimension): {}". format(bc_data.shape))

#%%

#   SPLIT DATA INTO FEATURES AND LABEL

bc_features = bc_data.copy() # COPY - Copy the data from db into features
bc_labels = bc_features.pop('diagnosis') # POP - DELETE THE SELECTED COLUMN


#%%

#   CHECK THE DATA
print("Features:\n", bc_features.head())
print("\n Label:\n", bc_labels.head())

#%%

#   ONE HOT ENCODE -LABEL-
bc_label_oh = pd.get_dummies(bc_labels) 

#   CHECK ONE HOT ENCODE -LABEL-
print(" ONE HOT ENCODE - LABEL - : \n ", bc_label_oh.head())

#%%

#   SPLIT THE FEATURES AND LABELS INTO TRAIN-VALIDATION - TEST SET SPLIT
from sklearn.model_selection import train_test_split

SEED=12345
x_train, x_iter, y_train, y_iter = train_test_split(bc_features,bc_label_oh,test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#%%
#   NORMALIZE THE FEATURES, 
#   FIT THE TRAINING DATA,

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_val = sc.transform(x_val)

#%%

#   CREATE A FEEDFORWARD NEURAL NETWORK USING TF KERAS - SEQUENTIAL API
import tensorflow as tf

num_input = x_train.shape[-1]
num_output = y_train.shape[-1]

model = tf.keras.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape = num_input))
#model.add(tf.keras.layers(input_shape=(bc_features.shape[-1],)))
model.add(tf.keras.layers.Dense(64, activation='elu'))
model.add(tf.keras.layers.Dense(32, activation='elu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_output, activation='softmax'))

#   COMPILE THE MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#%%
#   Train and evaluate model
#   Define callback functions: EarlyStopping and Tensorboard

base_log_path = r"C:\Users\HP\Desktop\Deep Learning with Python\Tensorboard"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)

EPOCHS = 100
BATCH_SIZE=32

history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb_callback,es_callback])


#%%

#   Evaluate with test data for wild testing

test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)

print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")


#%%
#   PREDICTIONs
#   NUMERICAL ENCODE USE ARGMAX
predictions_softmax = model.predict(x_test)
predictions = np.argmax(predictions_softmax, axis = -1)

y_test_element, y_test_idx = np.where(np.array(y_test) == 1)

for prediction, label in zip(predictions, np.array(y_test_idx)):
    print(f"prediction: {prediction} label: {label}")
#%%
