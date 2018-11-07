#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create pickled models from training data to use for model deployment website

Train download regression model and save with pickle as finalized_model.sav.
Train hours played regression model and save with pickle as finalized_model_hours.sav.

Created by: Kyle Dillon
Last updated: 10-7-18
"""

# In[]:

# Save Model Using Pickle
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
import pickle

dataframe = pd.read_csv('modeltrain.csv')
array = dataframe.values
X = array[:,0:16]
Y = array[:,16]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = linear_model.LinearRegression()
model.fit(X,Y)
# save the model to disk

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.score(X,Y))
print(array.shape)
print(X.shape)
print(Y.shape)

# In[]:

# Save Model Using Pickle
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
import pickle

dataframe = pd.read_csv('modelhours.csv')
array = dataframe.values
X = array[:,0:16]
Y = array[:,16]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = linear_model.LinearRegression()
model.fit(X,Y)
# save the model to disk

filename = 'finalized_model_hours.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.score(X,Y))
print(array.shape)
print(X.shape)
print(Y.shape)
