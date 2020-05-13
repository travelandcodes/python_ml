# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Data.csv');

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_X = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categories = X[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()

Y[: , 0] = labelencoder_Y.fit_transform(Y[: , 0])