# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:14:34 2017

@author: ACER
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#PART1 : DATA PREPROCESSING
# Importing the dataset
dataset = pd.read_csv(" Churn_Modelling.csv")

#It doesn't take into picture the upper limit column
#X stands for the independent variables
X = dataset.iloc[:, 3:13].values
                
#Y stands for the dependent variables
y = dataset.iloc[:, 13].values

# Encoding categorical data into numbers before splitting into test and train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#For France, Germany etc
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#For gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Getting rid of dummy variables
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#PART2: MAKING THE ANN
import keras
#Initializing the ANN
classifier = Sequential()

#Now we add the different layers such as the input layer. This has 11 independent variables
#I choose the Rectifier activation function for the hidden layers and the Sigmoid Activation Function for the output layer
#Take avg of number of nodes in the input and output layers as the number of nodes in the hidden layers which is (11+1)/2

#Add input and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer and set units to the number of categories to classify into.
#Dependent var with more than two categories: use soft max which is sigmoid for that. (ANN 7)
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN using stochastic descent
#Need to add an algorithm to compute weights
#If dep var has binary outcome, loss = "binary..", else if it has many categories loss = "categorical..."
classifier.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ["accuracy"])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#PART3: Making predictions
# Predicting the Test set results
y_pred = classifier.predict(X_test) 

#this returns true if the below statement is true else it returns false
y_pred = (y_pred > 0.5)
#We need to validate the model and see if it gives an accuracy on the test data as high as it gave for training set

#How to put a single observation in an array and figure out whether this person leaves or not
new_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#We obtained the same accuracy on both the train and test set!
#Yay, we have built our first ANN!

#PART3!
#Let us use k fold cross validation to evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
#Since KerasClassifier expects a function which returns the architecture of the ANN, we define a func
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1 )
mean = accuracies.mean()
variance = accuracies.std()

#PART 4: TUNING THE ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#Since KerasClassifier expects a function which returns the architecture of the ANN, we define a func
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer ,loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam','rmsprop'] }
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
