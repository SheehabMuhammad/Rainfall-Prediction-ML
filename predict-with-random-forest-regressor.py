import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import warnings; warnings.simplefilter('ignore')

#importing the dataset
dataset = pd.read_csv('data/rainfall_data_modified.csv')

#independent variable
X = dataset.iloc[:,0:3].values

#dependent variables
y = dataset.iloc[:,34].values

#preprocessing data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])

x = np.array(X)
y = np.array(y)

#Splitting and Fitting Data in our Model
from sklearn.model_selection import train_test_split
score_tracker = 0
for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15, random_state = i)
    regressor = RandomForestRegressor(n_estimators=200, max_depth = None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False)
    regressor.fit(X_train, y_train)

    predicted = regressor.predict(X_test)
    
    score_tracker += regressor.score(X_test,y_test)
    print("Progress:", i)

#average of performance
score_tracker = score_tracker/99;

print('Test score: {:.3f}'.format(score_tracker))

import matplotlib.pyplot as plt

plt.plot(y_test[0:100],label="Real")
plt.plot(predicted[0:100],color="red",label="Prediction")

plt.title('Rainfall Prediction with RandomForestRegressor',fontsize=14)
plt.xlabel('Observations')
plt.ylabel('Rainfall (millimeter)')

plt.legend()
plt.show()
