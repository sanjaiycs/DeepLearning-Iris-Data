#Importing all the required models

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#importing the iris dataset from sklearn.datasets
iris = load_iris()
print(iris)

# There are 3 Numerical dataset and 1 class label
X = iris.data         
y = iris.target        

#converting the class labeled data to numerical data
y = to_categorical(y, num_classes=3)

#spliting the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Defining the model's layers and activation functions
model = Sequential()  #create a simple FNN

model.add(Dense(16, activation='relu', input_shape=(4,))) #16 neurons and input feature is 4 and activation is Relu
model.add(Dense(12, activation='relu'))   #12 neurons activation is relu
model.add(Dense(3, activation='softmax')) #output layer- 3neurons ie 3 classes and activation is softmax

#Compling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Traing the model 
history = model.fit(X_train,y_train,epochs=100,batch_size=8)  #epochs is set to 100 ie the model can see the data 100 times & batch_size is set to 8 ie it updastes the weights every 8 samples

#Evaluating the model's performance and printing the accuracy score
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
