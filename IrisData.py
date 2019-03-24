# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 07:12:57 2019

@author: Aaditya
"""
#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

#Importing the dataset
dataset=pd.read_excel('Iris.xlsx')
dataset=dataset.dropna()
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1:]

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the data into Training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Histogram
dataset.hist()
plt.show()

#Scatter plots
scatter_matrix(dataset)
plt.show()

