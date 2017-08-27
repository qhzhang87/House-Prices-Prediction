#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression as lr, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, GridSearchCV, validation_curve
from sklearn.metrics import make_scorer, mean_squared_error

X_train = pd.read_pickle('final_x_train.pkl').values
y_train = pd.read_pickle('y_train.pkl').values
x_test = pd.read_pickle('final_x_test.pkl').values


#%%
def root_mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)**0.5


RMSE = make_scorer(root_mean_squared_error_, greater_is_better=False)


def param_test(model, paramgrid):
    # RMSE
    gsearch = GridSearchCV(
        estimator=model, param_grid=paramgrid, scoring=RMSE, cv=10, n_jobs=-1)
    gsearch.fit(X_train, y_train)
    print(gsearch.best_params_)
    print(-gsearch.best_score_)
    return gsearch.cv_results_, gsearch.best_params_, -gsearch.best_score_


def plotscore(testout, param_name):
    # plot cv test error vs. parameter
    y = -testout[0]['mean_test_score']
    x = testout[0]['params']
    x = [d[param_name] for d in x]
    plt.plot(x, y, 'o')


def get_score(model, param_name, param_range):
    # get cv training errors & testing errors
    train_scores, valid_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name,
        param_range,
        cv=10,
        scoring=RMSE,
        n_jobs=2)
    train_scores = -train_scores
    valid_scores = -valid_scores
    return {
        'training scores': train_scores,
        'validation scores': valid_scores,
        'param_grid': param_range,
        'best_score': valid_scores.mean(axis=1).min(),
        'best_param': param_range[valid_scores.mean(axis=1).argmin()],
        'training mean': np.mean(train_scores, axis=1),
        'training std': np.std(train_scores, axis=1),
        'validation mean': np.mean(valid_scores, axis=1),
        'validation std': np.std(valid_scores, axis=1)
    }


def plot_cv(train_scores, valid_scores, param_range):
    # plot cv training/test error vs. parameter
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    lw = 2
    plt.semilogx(
        param_range,
        train_scores_mean,
        label="Training score",
        color="darkorange")
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange")
    plt.semilogx(
        param_range,
        valid_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw)
    plt.fill_between(
        param_range,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw)
    plt.legend(loc="best")


#%% Ridge
# param_test(Ridge(), {'alpha': np.linspace(15, 25, 11)})
param_test(Ridge(), {'alpha': np.linspace(1, 4, 21)})

#%% Lasso
# param_test(Lasso(max_iter=3000), {'alpha': np.linspace(0.0001, 0.001, 10)})
param_test(Lasso(max_iter=3000), {'alpha': np.logspace(-4, -0.5, 50)})
param_test(Lasso(max_iter=2500), {'alpha': np.linspace(.0002, 0.0003, 11)})

#%% Random Forest
model1 = RandomForestRegressor(n_estimators=200, random_state=8)
test1 = param_test(model1, {'max_features': [.1, .3, .5, .7, .9]})
plotscore(test1, 'max_features')

param1 = {
    'max_depth': [15, 20, 25, 30],
    'max_features': np.linspace(0.2, 0.4, 9)
}
test3 = param_test(model1, param1)
plotscore(test3, 'max_depth')
plotscore(test3, 'max_features')

model2 = RandomForestRegressor(
    max_depth=20, max_features=0.375, random_state=8)
rf1 = get_score(model2, "n_estimators", range(10, 60, 10))
plot_cv(rf1['training scores'], rf1['validation scores'], rf1['param_grid'])
rf1

#%% Bagging
# Ridge
model1 = BaggingRegressor(base_estimator=Ridge(alpha=2.2), random_state=8)
param1 = {'n_estimators': range(20, 251, 10)}
bag1 = param_test(model1, param1)
plotscore(bag1, 'n_estimators')
# Lasso
model1 = BaggingRegressor(base_estimator=Lasso(), random_state=8)
param1 = {'n_estimators': range(10, 201, 20)}
bag1 = param_test(model1, param1)
plotscore(bag1, 'n_estimators')

#%% Boosting
model1 = AdaBoostRegressor(
    base_estimator=Lasso(alpha=0.00044, max_iter=3000), random_state=8)
param1 = {'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50]}
ada1 = param_test(model1, param1)
plotscore(ada1, 'n_estimators')

#%% XGBoost
param1 = {'max_depth': range(1, 7, 2), 'min_child_weight': range(1, 6, 2)}
model1 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=500,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8)
param_test(model1, param1)

param2 = {'max_depth': [2, 3, 4], 'min_child_weight': [4, 5, 6]}
param_test(model1, param2)

param3 = {'gamma': [i / 10.0 for i in range(0, 5)]}
model2 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=500,
    max_depth=3,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.75)
param_test(model2, param3)

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
model3 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=500,
    max_depth=3,
    min_child_weight=5,
    gamma=0)
param_test(model3, param_test4)

param_test5 = {'subsample': [.75, .8, .85], 'colsample_bytree': [.75, .8, .85]}
g5 = param_test(model3, param_test5)

model4 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=500,
    max_depth=3,
    min_child_weight=5,
    gamma=0,
    subsample=.75,
    colsample_bytree=.8)

param_test6 = {'reg_alpha': [0, 1e-4, 1e-2, 0.1, 1]}
g6 = param_test(model4, param_test6)

param_test7 = {'reg_alpha': [0, 1e-5]}
g7 = param_test(model4, param_test7)

param_test8 = {'reg_alpha': [0.01, 0.015]}
g8 = param_test(model4, param_test8)

param_test9 = {'reg_lambda': [0.1, 1, 10]}
g9 = param_test(model4, param_test9)

param_test10 = {'reg_lambda': [0.5, 1, 1.5]}
g10 = param_test(model4, param_test10)

param_test11 = {'reg_lambda': np.linspace(0), 'reg_alpha': np.linspace(1)}
g11 = param_test(model4, param_test11)

model5 = XGBRegressor(  #learning_rate =0.1, n_estimators=500, 
    max_depth=3,
    min_child_weight=5,
    gamma=0,
    subsample=.75,
    colsample_bytree=.8,
    reg_alpha=0,
    reg_lambda=1)

param_test12 = {
    'n_estimators': [200, 300, 400, 500, 600, 700, 800],
    'learning_rate': [0.05, 0.1]
}

g12 = param_test(model5, param_test12)
xgbout(g12)