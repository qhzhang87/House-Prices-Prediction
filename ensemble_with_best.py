#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble with best models
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression as lr

x_train = pd.read_pickle('final_x_train.pkl')
x_test = pd.read_pickle('final_x_test.pkl')
y_train = pd.read_pickle('y_train.pkl')


def create_submission(prediction, score):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # sub_file = 'prediction_training.csv'
    print('Creating submission: ', sub_file)
    pd.DataFrame({
        'Id': x_test['Id'].values,
        'SalePrice': prediction
    }).to_csv(
        sub_file, index=False)


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions)**0.5


RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, train, test, ytr):
        X = train.values
        y = ytr.values
        T = test.values
        folds = list(
            KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=0))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros(
            (T.shape[0],
             len(self.base_models)))  # X need to be T when do test prediction
        for i, reg in enumerate(base_models):
            print("Fitting the base model...")
            S_test_i = np.zeros(
                (T.shape[0],
                 len(folds)))  # X need to be T when do test prediction
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = reg.predict(T)[:]
            #    S_test_i[:,j] = reg.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        print("Stacking base models...")
        param_grid = {
            'alpha': [
                1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.8, 1e0, 3,
                5, 7, 1e1, 2e1, 5e1
            ],
        }
        grid = GridSearchCV(
            estimator=self.stacker,
            param_grid=param_grid,
            n_jobs=1,
            cv=5,
            scoring=RMSE)
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
        except:
            pass

        y_pred = grid.predict(S_test)[:]
        return y_pred, -grid.best_score_


base_models = [
    XGBRegressor(
        learning_rate=0.05,
        n_estimators=600,
        max_depth=3,
        min_child_weight=3,
        gamma=0,
        subsample=.7,
        colsample_bytree=.9,
        reg_alpha=0,
        reg_lambda=1),
    Lasso(alpha=0.00044),
    Ridge(alpha=16.8),
    BaggingRegressor(
        n_estimators=240,
        base_estimator=Ridge(16.8, max_iter=2000),
        random_state=8),
    BaggingRegressor(
        n_estimators=270,
        base_estimator=Lasso(alpha=0.00036, max_iter=2000),
        random_state=8),
    RandomForestRegressor(
        max_depth=20, max_features=0.375, n_estimators=50, random_state=8)
]

ensem = ensemble(n_folds=10, stacker=lr(), base_models=base_models)

S_train, S_test, y_pred, score = ensem.fit_predict(x_train, x_test, y_train)

a = lr().fit(S_train, y_train)
a.coef_

create_submission(np.exp(y_pred), score)
