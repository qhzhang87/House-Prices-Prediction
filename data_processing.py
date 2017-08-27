#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing
"""
#%%
import numpy as np
import pandas as pd
from scipy.stats import skew  # , skewtest
import importlib


def convert_dtypes(df, features):
    '''
    convert feature dtype from numerical to categorical

    Parameters
    ----------
    df : dataframe
        dataframe
    features : list
        list of categorical features that are read as numerical
    '''
    df[features] = df[features].astype(str)


def na_to_cate(df, features, cate_name):
    '''
    some features may have NA as a category,
    we need to convert them back from nan to str

    Parameters
    ----------
    df : dataframe
        dataframe
    features : list
        list of features that contain nan as a category
    cate_name: str
        the category name used to replace nan
    Returns
    -------
    df : dataframe
        dataframe
    '''
    df[features] = df[features].fillna(cate_name)
    return df


def fill_missing_value(df, num_features, num_method, cate_features):
    '''
    fill missing values

    Parameters
    ----------
    df : dataframe
        dataframe before
    num_features : list
        list of numeric features with missing values that needs to be filled
    num_method : 'mean' or 'median'
        fill with mean or median
    cate_features : list
        list of categorical features with missing values that needs to be filled
    Returns
    -------
    df: dataframe
        dataframe after
    '''
    # For numerical features, replace with mean or median
    if num_method == 'mean':
        df = df.fillna(df.mean()[num_features])
    elif num_method == 'median':
        df = df.fillna(df.median()[num_features])
    else:
        print('Please enter the right method')
        return
    # For categorical features, replace with the category that occurs the most
    for i in cate_features:
        df[i] = df[i].fillna(df[i].value_counts().nlargest(1).index.values[0])

    return df


def drop_outlier(df, outlier_idx):
    '''
    drop outliers

    Parameters
    ----------
    df : dataframe
        dataframe
    outlier_idx : list
        list of outlier observations' index
    Returns
    -------
    df : dataframe
        dataframe
    '''
    df = df.drop(outlier_idx)
    return df


def drop_features(df, features):
    '''
    drop features from the dataframe

    Parameters
    ----------
    df : dataframe
        dataframe
    features : list
        list of features that will be dropped
    Returns
    -------
    df : dataframe
        dataframe
    '''
    df = df.drop(features, axis=1)
    return df


def scale_data(df):
    '''
    standardlize numerical features

    Parameters
    ----------
    df : dataframe
        dataframe
    '''
    numeric_col = df.select_dtypes(exclude=['O']).columns
    numeric_col_mean = df.loc[:, numeric_col].mean()
    numeric_col_std = df.loc[:, numeric_col].std()
    df.loc[:, numeric_col] = (
        df.loc[:, numeric_col] - numeric_col_mean) / numeric_col_std


#%% Load data
train_df = pd.read_csv('./input/train.csv', index_col=0)  # encoding='gb18030'
test_df = pd.read_csv('./input/test.csv', index_col=0)

# If a file of data types is given, we can specify the data types when loading the data
# e.g.  data_types = pd.read_excel('', index_col=0)
#       data_types = data_types[data_types.变量类型=='object'].iloc[1:].变量类型.to_dict()
#       df = pd.read_csv('', dtype=data_types)

#%% Normalize the label
# use log transformation
y_train = np.log(train_df.pop('SalePrice'))

#%% Correct the dtypes of features whose dtypes are wrongly readed
convert_dtypes(train_df, ['MSSubClass', 'MoSold'])
convert_dtypes(test_df, ['MSSubClass', 'MoSold'])

#%% For some features, NA in fact means 'no' category, they are not missing values
na_cols = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
]

train_df = na_to_cate(train_df, na_cols, 'NA')
test_df = na_to_cate(test_df, na_cols, 'NA')

#%% Fill missing values
# list of features that contain missing values
trainNa = train_df.isnull().sum()[train_df.isnull().sum() > 0].sort_values(
    ascending=False).index
testNa = test_df.isnull().sum()[test_df.isnull().sum() > 0].sort_values(
    ascending=False).index

# Those variables that in fact have NAs are:
#   Train:
#       Categorical:  MasVnrType, Electrical
#       Numerical: LotFrontage, GarageYrBlt, MasVnrArea
#   Test:
#       Categorical:  MasVnrType
#               | MSZoning, Utilities ,Functional ,Exterior2nd, SaleType,
#                   Exterior1st, KitchenQual
#       Numerical: LotFrontage, GarageYrBlt, MasVnrArea
#               | BsmtHalfBath, BsmtFullBath,BsmtFinSF2, BsmtFinSF1,
#                   BsmtUnfSF,TotalBsmtSF, GarageArea, GarageCars

train_df = fill_missing_value(train_df,
                              ['LotFrontage', 'GarageYrBlt', 'MasVnrArea'],
                              'median', ['MasVnrType', 'Electrical'])
test_df = fill_missing_value(test_df, [
    'LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath',
    'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea',
    'GarageCars'
], 'median', [
    'MasVnrType', 'MSZoning', 'Functional', 'Utilities', 'Exterior2nd',
    'SaleType', 'Exterior1st', 'KitchenQual'
])

#%% Drop outlier
# drop all > 4000 ??? 1299,  524, 692, 1183
train_df = drop_outlier(train_df, [524, 1299])
y_train = drop_outlier(y_train, [524, 1299])

# %%
train_df.to_pickle('x_train.pkl')
test_df.to_pickle('x_test.pkl')
y_train.to_pickle('y_train.pkl')

#%%
# -> feature engineering.py
fe = importlib.import_module('feature engineering')
x_all = fe.x_all

#%%
x_train = x_all.loc[train_df.index]
x_test = x_all.loc[test_df.index]

#%% Transformations
# PS: log(p+1) transformations are not the available transformations for reducing skewness.
# I have tried sqrt, which yields better results as compared to log(p+1)
# (for certain features only. log(1+p) is a better way to go for this dataset at least)
numeric_cols = x_train.columns[x_train.dtypes != 'object']

skewed_feats = x_train[numeric_cols].apply(
    lambda x: skew(x.dropna()))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

x_train[skewed_feats] = np.log1p(x_train[skewed_feats])
x_test[skewed_feats] = np.log1p(x_test[skewed_feats])

#%% Standardlize numerical features
scale_data(x_train)
scale_data(x_test)

#%% Categorical variables to dummy variables
x_all_dummy = pd.get_dummies(pd.concat((x_train, x_test), axis=0))
x_train = x_all_dummy.loc[x_train.index]
x_test = x_all_dummy.loc[x_test.index]

# dummy_train_df = pd.get_dummies(train_df, drop_first = True)
# dummy_test_df = pd.get_dummies(test_df, drop_first = True)

#%%
x_train.to_pickle('final_x_train.pkl')
x_test.to_pickle('final_x_test.pkl')

#%%
## delete features with a lot of NAs
#to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
#
## delete multicolinearity
#to_delete = ['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd']
#
## delete ??
#'LowQualFinSF''1stFlrSF''2ndFlrSF'
#del_tdummy_train_df = tdummy_train_df.drop(to_delete,axis=1)
#del_tdummy_test_df = tdummy_test_df.drop(to_delete,axis=1)
#
#
#all_data = all_df.drop(to_delete,axis=1)