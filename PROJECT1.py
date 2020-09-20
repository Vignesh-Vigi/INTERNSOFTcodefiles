# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 09:23:14 2020

@author: Vignesh
"""

#PROJECT 1.


#Importing the libraries.
import pandas as pd
import matplotlib.pyplot as plt


#Reading the data from hte files.
data = pd.read_csv('ADVERTISING.csv')
data.head()


#Visualizing the data set.
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(20,10))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])


#Creating X&Y for linear regression.
feature_cols = ['TV']
X = data[feature_cols]
Y = data.Sales


#Importing LINEAR REGRESSION Algorithm.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
print(lr.intercept_)
print(lr.coef_)

ans = 6.9748214882298925 + 0.05546477*50
print(ans)


#Create a dataframe wiht min and max value of table.
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

Preds = lr.predict(X_new)
Preds

data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, Preds, c='red', linewidth=1)

import statsmodels.formula.api as smf
lr = smf.ols(formula='Sales ~ TV', data=data).fit()
lr.conf_int()


#Finding probability values.
lr.pvalues


#Finding the r-squared Values
lr.rsquared


#Multi linear regression.
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
Y = data.Sales
lr = LinearRegression()
lr.fit(X,Y)

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()

lm.rsquared

lm = smf.ols(formula='Sales ~ TV+Radio',data=data).fit()
lm.conf_int()
lm.summary()

lm.rsquared


#THE END