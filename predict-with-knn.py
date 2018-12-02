# KNN regression - 1 
# applied in Bangladesh rainfall data

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import warnings; warnings.simplefilter('ignore') # Jupyter notebook warning message remove

#importing the dataset
dataset = pd.read_csv('data/rainfall_data_modified.csv')

#independent variable
X = dataset.iloc[:,0:3].values # considering all station name

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
    
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.15, random_state = i)
    
    print('\n', X_train, '\n')
    
    kNeighborsRegressor = KNeighborsRegressor(n_neighbors = 6)
    kNeighborsRegressor.fit(X_train, y_train)
    predicted = kNeighborsRegressor.predict(X_test)
    
    #comulative sum of each test score
    score_tracker += kNeighborsRegressor.score(X_test,y_test)

#average of performance
score_tracker = score_tracker/99;

print('Test score: {:.3f}'.format(score_tracker))

import matplotlib.pyplot as plt

plt.plot(y_test[0:100],label="Real")
plt.plot(predicted[0:100],color="red",label="Prediction")

plt.title('Rainfall Prediction with KNN Regressor',fontsize=14)
plt.xlabel('Observations')
plt.ylabel('Rainfall (millimeter)')

plt.legend()
plt.show()
