import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sn


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255 #normalizing data to values between 0 and 1
X_test = X_test/255

X_train = X_train.reshape(len(X_train),28*28) #flattening data to a 784 length array
X_test = X_test.reshape(len(X_test),28*28)

model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,), activation='sigmoid')])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5) #training data using compiled model

model.evaluate(X_test,y_test) #evaluate accuracy of model: 92.54%

y_predicted = model.predict(X_test) #generates array of predictions from model   
y_predicted_labels = [np.argmax(i) for i in y_predicted]

matrix = tf.math.confusion_matrix(labels=y_test, predictions = y_predicted_labels) #creating confusion matrix to compare predictions to true values

plt.figure(figsize = (10,7)) #plotting confusion matrix using seaborn
sn.heatmap(matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth') 
plt.show()



                         
