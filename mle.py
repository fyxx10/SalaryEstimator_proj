#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mensahotooemmanuel
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv('eda_data1.csv')

# choose relevant columns 
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

# get dummy data 
df_dum = pd.get_dummies(df_model)

# train test split 
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression 
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

# cross validation for multiple linear regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# show the best alpha value
best_alpha = df_err.loc[df_err['error'].idxmax()]['alpha']
print("Best alpha:", best_alpha)

# confirm and show the alpha value that resulted in the lowest negative mean absolute error during cross-validation
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion': ('squared_error', 'absolute_error'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

#show the best score
gs.best_score_
# to show the best model
gs.best_estimator_


from sklearn.utils import check_random_state

# random seed for NumPy (for consistency in data splitting)
np.random.seed(42)

# random seed for scikit-learn (for consistent model initialization)
random_state = check_random_state(42)

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm) #overfitting thus, applying the lasso regularization
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

# using Lasso regularization on the linear regression model

# Assuming lm is your linear regression model
lmR = Lasso(alpha=0.01)  
lmR.fit(X_train, y_train)
tpred_lmR = lasso.predict(X_test)
mean_absolute_error(y_test, tpred_lmR)


mean_absolute_error(y_test,(tpred_lmR+tpred_rf)/2)

