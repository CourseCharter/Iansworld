# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np

data = pd.read_csv('../data/glassdoor.csv')

toprows = data.head()

#cleaning the data




#see relations on various charts
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath


#saving charts to png
projects = str(PureWindowsPath('C:/Users/user/projects/iansworld'))
fig.savefig(projects + '\static\\skills\\skills_model')

#look at each features' correlation to survival
#corr_table = data.corr().abs()[[""]]

#drop insignifant


#split the train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[['sex','pclass','age','relatives','fare']], data.survived, test_size=0.2, random_state=0)

#normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#classifying 
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

nn_summary = model.summary()

#complining
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=50)

#evaluate
from sklearn import metrics
y_pred = model.predict_classes(X_test)
nn_results = metrics.accuracy_score(y_test, y_pred)

#pickling
import pickle
from sklearn.externals import joblib
filename= 'jobskills_model.pkl'
joblib.dump(model, filename)



