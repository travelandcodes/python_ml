import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan , strategy = "mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()

#X[: , 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder()
spare=onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()

dataset.Country.values.reshape(-1,1)

columns = ["Country_"+str(int(i)) for i in range(dataset.shape[1]-1)]

dfOneHot= pd.DataFrame(spare,columns=columns)
df = pd.concat([dataset, dfOneHot], axis=1)

df= df.drop(['Country'], axis=1)
# to drop the Country column 
df