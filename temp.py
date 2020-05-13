import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Data.csv')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# filling the nan values with themean of the respective column
dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())
dataset['Salary']=dataset['Salary'].fillna(dataset['Salary'].mean())

# instantiating the OneHotEncoder
onehotencoder = OneHotEncoder()

#fitting the data
spare=onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()
dataset.Country.values.reshape(-1,1)

#generating column names
columns = ["Country_"+str(int(i)) for i in range(dataset.shape[1]-1)]

#creating a new dataframe for the one hot encoded data
dfOneHot= pd.DataFrame(spare,columns=columns)

#concatenating the two dataframes
df = pd.concat([dataset, dfOneHot], axis=1)

# removing the country column from the final viz
df= df.drop(['Country'], axis=1)

# calling the final output
df
