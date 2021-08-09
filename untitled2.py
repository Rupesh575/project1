# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:44:11 2021

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

dataset=pd.read_csv('E:/Weather.csv/Summary of Weather.csv')
dataset.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()
x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)
x_test,x_pred,y_test,y_pred=train_test_split(x,y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(x_pred,y_pred)
print(regressor.intercept_)
print(regressor.coef_)
y_pred=regressor.predict(x_test)
df=pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})
print(df)
plt.scatter(x_test,y_test,color='gray')
plt.plot(x_test,y_pred,color='red',linewidth=2,)
plt.show()
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print('fitting score:',r2)
