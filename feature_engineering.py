#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering
"""

import pandas as pd
import numpy as np
# from geopy.geocoders import Nominatim

#%% Load data
x_train = pd.read_pickle('x_train.pkl')
x_test = pd.read_pickle('x_test.pkl')

x_all = pd.concat((x_train, x_test), axis=0)

#%% convert neighborhood to long & lat
# def get_latlng(df, address_col, address_map=None):
#     '''
#     get latitude and longitude from address

#     Parameters
#     ----------
#     df : dataframe
#         dataframe before
#     address_col : string
#         column name that represents the address
#     address_map: dict
#         optional, mapping of address_col values to real address
#     Returns
#     -------
#     df: dataframe
#         dataframe after
#     '''
#     lat_dict = {}
#     lng_dict = {}
#     geolocator = Nominatim()

#     if address_map is None:
#         addresses = df.address_col.unique()
#         for i in addresses:
#             location = geolocator.geocode(i)
#             if location is None:
#                 print('No geo info for address: %s' % i)
#                 continue
#             lat_dict[i] = location.latitude
#             lng_dict[i] = location.longitude
#     else:
#         for key, value in address_map.items():
#             location = geolocator.geocode(value)
#             if location is None:
#                 print('No geo info for address: %s' % key)
#                 continue
#             lat_dict[key] = location.latitude
#             lng_dict[key] = location.longitude

#     df['lat'] = df[address_col].replace(lat_dict)
#     df['lng'] = df[address_col].replace(lng_dict)

# get_latlng(x_all, 'Neighborhood', neighborhood_dict)

#%%  1. Location based on the school location in the district ??
# neighborhood doesn't choosen by xgboost
x_all["lat"] = x_all.Neighborhood.replace({
    'Blmngtn': 42.062806,
    'Blueste': 42.009408,
    'BrDale': 42.052500,
    'BrkSide': 42.033590,
    'ClearCr': 42.025425,
    'CollgCr': 42.021051,
    'Crawfor': 42.025949,
    'Edwards': 42.022800,
    'Gilbert': 42.027885,
    'GrnHill': 42.000854,
    'IDOTRR': 42.019208,
    'Landmrk': 42.044777,
    'MeadowV': 41.991866,
    'Mitchel': 42.031307,
    'NAmes': 42.042966,
    'NoRidge': 42.050307,
    'NPkVill': 42.050207,
    'NridgHt': 42.060356,
    'NWAmes': 42.051321,
    'OldTown': 42.028863,
    'SWISU': 42.017578,
    'Sawyer': 42.033611,
    'SawyerW': 42.035540,
    'Somerst': 42.052191,
    'StoneBr': 42.060752,
    'Timber': 41.998132,
    'Veenker': 42.040106
})

x_all["lon"] = x_all.Neighborhood.replace({
    'Blmngtn': -93.639963,
    'Blueste': -93.645543,
    'BrDale': -93.628821,
    'BrkSide': -93.627552,
    'ClearCr': -93.675741,
    'CollgCr': -93.685643,
    'Crawfor': -93.620215,
    'Edwards': -93.663040,
    'Gilbert': -93.615692,
    'GrnHill': -93.643377,
    'IDOTRR': -93.623401,
    'Landmrk': -93.646239,
    'MeadowV': -93.602441,
    'Mitchel': -93.626967,
    'NAmes': -93.613556,
    'NoRidge': -93.656045,
    'NPkVill': -93.625827,
    'NridgHt': -93.657107,
    'NWAmes': -93.633798,
    'OldTown': -93.615497,
    'SWISU': -93.651283,
    'Sawyer': -93.669348,
    'SawyerW': -93.685131,
    'Somerst': -93.643479,
    'StoneBr': -93.628955,
    'Timber': -93.648335,
    'Veenker': -93.657032
})

#%% 2. Quality categorical variable to numeric variable??
# ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
for i in ['ExterQual', 'ExterCond', 'heatingQC', 'KitchenQual']:
    x_all = x_all.replace({i: {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}})

for i in ['BsmtQual', 'fireplaceQu', 'GarageQual', 'GarageCond']:
    x_all = x_all.replace({
        i: {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 0
        }
    })

x_all = x_all.replace({
    'PoolQC': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'NA': 0
    }
})

x_all = x_all.replace({
    'BsmtExposure': {
        'Gd': 4,
        'Av': 3,
        'Mn': 2,
        'No': 1,
        'NA': 0
    }
})

x_all = x_all.replace({
    'BsmtFinType1': {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
        'NA': 0
    }
})

x_all = x_all.replace({
    'BsmtFinType2': {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
        'NA': 0
    }
})

#%% 3. Numerical To categorical
x_all['ifOpenPorch'] = np.where(x_all['OpenPorchSF'] == 0, 0, 1)
x_all['ifEnclosedPorch'] = np.where(x_all['EnclosedPorch'] == 0, 0, 1)
x_all['if3SsnPorch'] = np.where(x_all['3SsnPorch'] == 0, 0, 1)
x_all['ifScreenPorch'] = np.where(x_all['ScreenPorch'] == 0, 0, 1)
x_all['if2ndFlr'] = np.where(x_all['2ndFlrSF'] == 0, 0, 1)
x_all['ifLowQualFin'] = np.where(x_all['LowQualFinSF'] == 0, 0, 1)
x_all['ifWoodDeck'] = np.where(x_all['WoodDeckSF'] == 0, 0, 1)

x_all['ifBstmAllFin'] = np.where(
    (x_all['TotalBsmtSF'] != 0) & (x_all['BsmtUnfSF'] == 0), 1, 0)

# If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
x_all["Remodeled"] = (x_all["YearRemodAdd"] != x_all["YearBuilt"]) * 1
# Did a remodeling happen in the year the house was sold?
x_all["RecentRemodel"] = (x_all["YearRemodAdd"] == x_all["YrSold"]) * 1
# Was this house sold in the year it was built?
x_all["VeryNewHouse"] = (x_all["YearBuilt"] == x_all["YrSold"]) * 1

#%% 4. Combination of features
# Overall quality of the house
x_all["OverallGrade"] = x_all["OverallQual"] * x_all["OverallCond"]
# Overall quality of the garage
x_all["GarageGrade"] = x_all["GarageQual"] * x_all["GarageCond"]
# Overall quality of the exterior
x_all["ExterGrade"] = x_all["ExterQual"] * x_all["ExterCond"]
# Overall quality of the basement
x_all["BsmtGrade"] = x_all["BsmtQual"] * x_all["BsmtCond"]

# Overall kitchen score (size\number * quality)
x_all["KitchenScore"] = x_all["KitchenAbvGr"] * x_all["KitchenQual"]
# Overall fireplace score
x_all["FireplaceScore"] = x_all["Fireplaces"] * x_all["FireplaceQu"]
# Overall garage score
x_all["GarageScore"] = x_all["GarageArea"] * x_all["GarageQual"]
# Overall pool score
x_all["PoolScore"] = x_all["PoolArea"] * x_all["PoolQC"]
# Overall basement score
x_all["BsmtScore1"] = x_all["BsmtFinSF1"] * x_all["BsmtFinType1"]
x_all["BsmtScore2"] = x_all["BsmtFinSF2"] * x_all["BsmtFinType2"]

# Total number of bathrooms
x_all["TotalBath"] = x_all["BsmtFullBath"] + (
    0.5 * x_all["BsmtHalfBath"]) + x_all["FullBath"] + (
        0.5 * x_all["HalfBath"])
# Total SF for house (incl. basement)
x_all["AllSF"] = x_all["GrLivArea"] + x_all["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
x_all["AllFlrsSF"] = x_all["1stFlrSF"] + x_all["2ndFlrSF"]
# Total SF for porch
x_all[
    "AllPorchSF"] = x_all["OpenPorchSF"] + x_all["EnclosedPorch"] + x_all["3SsnPorch"] + x_all["ScreenPorch"]

# Basement bathrooms per square feet
x_all['BsmtFullBathRate'] = x_all['BsmtFullBath'] / x_all['TotalBsmtSF']
x_all['BsmtHalfBathRate'] = x_all['BsmtHalfBath'] / x_all['TotalBsmtSF']
# here, TotalBsmtSF may equal to 0
x_all.loc[x_all['TotalBsmtSF'] == 0, 'BsmtFullBathRate'] = 0
x_all.loc[x_all['TotalBsmtSF'] == 0, 'BsmtHalfBathRate'] = 0
#x_all['BsmtFullBathRate'] = x_all['BsmtFullBathRate'].replace(np.NaN, 0)
#x_all['BsmtHalfBathRate'] = x_all['BsmtHalfBathRate'].replace(np.NaN, 0)

# bathrooms above grade per square feet
x_all['FullBathRate'] = x_all['FullBath'] / x_all['GrLivArea']
x_all['HalfBathRate'] = x_all['HalfBath'] / x_all['GrLivArea']
# kitchen number above grade per square feet
x_all['KitchenRate'] = x_all['KitchenAbvGr'] / x_all['GrLivArea']
# bedroom number above grade per square feet
x_all['BedroomRate'] = x_all['BedroomAbvGr'] / x_all['GrLivArea']
# total rooms above grade per sf
x_all['TotRmsRate'] = x_all['TotRmsAbvGrd'] / x_all['GrLivArea']
# Fireplaces per sf
x_all['FireplaceRate'] = x_all['Fireplaces'] / x_all['GrLivArea']

# bathrooms per bedroom above grade
x_all['BathPerBed'] = (
    x_all['FullBath'] + .5 * x_all['HalfBath']) / x_all['BedroomAbvGr']
# here, 'BedroomAbvGr' may equal to 0
x_all.loc[x_all['BedroomAbvGr'] == 0, 'BathPerBed'] = 0

# MasVnrType: Masonry veneer type
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
# MasVnrArea: Masonry veneer area in square feet

# MiscFeature: Miscellaneous feature not covered in other categories
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
# MiscVal: $Value of miscellaneous feature

#%% More new features
# IR2 and IR3 don't appear that often, so just make a distinction
# between regular and irregular.
#x_all["IsRegularLotShape"] = (x_all["LotShape"] == "Reg") * 1

# Most properties are level; bin the other possibilities together
# as "not level".
# x_all["IsLandLevel"] = (x_all["LandContour"] == "Lvl") * 1

# Most land slopes are gentle; treat the others as "not gentle".
# x_all["IsLandSlopeGentle"] = (x_all["LandSlope"] == "Gtl") * 1

# Most properties use standard circuit breakers.
#all_df["IsElectricalSBrkr"] = (all_df["Electrical"] == "SBrkr") * 1

# About 2/3rd have an attached garage.
#all_df["IsGarageDetached"] = (all_df["GarageType"] == "Detchd") * 1

# Most have a paved drive. Treat dirt/gravel and partial pavement
# as "not paved".
#all_df["IsPavedDrive"] = (all_df["PavedDrive"] == "Y") * 1

# The only interesting "misc. feature" is the presence of a shed.
#all_df["HasShed"] = (all_df["MiscFeature"] == "Shed") * 1.

## Months with the largest number of deals may be significant
#all_df["HighSeason"] = all_df["MoSold"].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

x_all["NewerDwelling"] = x_all["MSSubClass"].replace({
    20: 1,
    30: 0,
    40: 0,
    45: 0,
    50: 0,
    60: 1,
    70: 0,
    75: 0,
    80: 0,
    85: 0,
    90: 0,
    120: 1,
    150: 0,
    160: 1,
    180: 0,
    190: 0
})

#%% Create new features (more)
# 1* Simplifications of existing features
# all_df["SimplOverallQual"] = all_df.OverallQual.replace({
#     1: 1,
#     2: 1,
#     3: 1,  # bad
#     4: 2,
#     5: 2,
#     6: 2,  # average
#     7: 3,
#     8: 3,
#     9: 3,
#     10: 3  # good
# })
# all_df["SimplOverallCond"] = all_df.OverallCond.replace({
#     1: 1,
#     2: 1,
#     3: 1,  # bad
#     4: 2,
#     5: 2,
#     6: 2,  # average
#     7: 3,
#     8: 3,
#     9: 3,
#     10: 3  # good
# })
# all_df["SimplPoolQC"] = all_df.PoolQC.replace({
#     1: 1,
#     2: 1,  # average
#     3: 2,
#     4: 2  # good
# })
# all_df["SimplGarageCond"] = all_df.GarageCond.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplGarageQual"] = all_df.GarageQual.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplFunctional"] = all_df.Functional.replace({
#     1: 1,
#     2: 1,  # bad
#     3: 2,
#     4: 2,  # major
#     5: 3,
#     6: 3,
#     7: 3,  # minor
#     8: 4  # typical
# })
# all_df["SimplKitchenQual"] = all_df.KitchenQual.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplHeatingQC"] = all_df.HeatingQC.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["BadHeating"] = all_df.HeatingQC.replace({
#     1: 1,  # bad
#     2: 0,
#     3: 0,  # average
#     4: 0,
#     5: 0  # good
# })
# all_df["SimplBsmtFinType1"] = all_df.BsmtFinType1.replace({
#     1: 1,  # unfinished
#     2: 1,
#     3: 1,  # rec room
#     4: 2,
#     5: 2,
#     6: 2  # living quarters
# })
# all_df["SimplBsmtFinType2"] = all_df.BsmtFinType2.replace({
#     1: 1,  # unfinished
#     2: 1,
#     3: 1,  # rec room
#     4: 2,
#     5: 2,
#     6: 2  # living quarters
# })
# all_df["SimplBsmtCond"] = all_df.BsmtCond.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplBsmtQual"] = all_df.BsmtQual.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplExterCond"] = all_df.ExterCond.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })
# all_df["SimplExterQual"] = all_df.ExterQual.replace({
#     1: 1,  # bad
#     2: 1,
#     3: 1,  # average
#     4: 2,
#     5: 2  # good
# })

# taken from https://www.kaggle.com/yadavsarthak/house-prices-advanced-regression-techniques/you-got-this-feature-engineering-and-lasso
# all_df['1stFlr_2ndFlr_Sf'] = np.log1p(all_df['1stFlrSF'] + all_df['2ndFlrSF'])
# all_df['All_Liv_SF'] = np.log1p(all_df['1stFlr_2ndFlr_Sf'] + all_df['LowQualFinSF'] + all_df['GrLivArea'])

#%%
# 3* Polynomials on the top 10 existing features
# all_df["OverallQual-s2"] = all_df["OverallQual"]**2
# all_df["OverallQual-s3"] = all_df["OverallQual"]**3
# all_df["OverallQual-Sq"] = np.sqrt(all_df["OverallQual"])

# all_df["AllSF-2"] = all_df["AllSF"]**2
# all_df["AllSF-3"] = all_df["AllSF"]**3
# all_df["AllSF-Sq"] = np.sqrt(all_df["AllSF"])

# all_df["AllFlrsSF-2"] = all_df["AllFlrsSF"]**2
# all_df["AllFlrsSF-3"] = all_df["AllFlrsSF"]**3
# all_df["AllFlrsSF-Sq"] = np.sqrt(all_df["AllFlrsSF"])

# all_df["SimplOverallQual-s2"] = all_df["SimplOverallQual"]**2
# all_df["SimplOverallQual-s3"] = all_df["SimplOverallQual"]**3
# all_df["SimplOverallQual-Sq"] = np.sqrt(all_df["SimplOverallQual"])

# all_df["GrLivArea-2"] = all_df["GrLivArea"]**2
# all_df["GrLivArea-3"] = all_df["GrLivArea"]**3
# all_df["GrLivArea-Sq"] = np.sqrt(all_df["GrLivArea"])

# all_df["GarageCars-2"] = all_df["GarageCars"]**2
# all_df["GarageCars-3"] = all_df["GarageCars"]**3
# all_df["GarageCars-Sq"] = np.sqrt(all_df["GarageCars"])

# all_df["ExterQual-2"] = all_df["ExterQual"]**2
# all_df["ExterQual-3"] = all_df["ExterQual"]**3
# all_df["ExterQual-Sq"] = np.sqrt(all_df["ExterQual"])

# all_df["TotalBath-2"] = all_df["TotalBath"]**2
# all_df["TotalBath-3"] = all_df["TotalBath"]**3
# all_df["TotalBath-Sq"] = np.sqrt(all_df["TotalBath"])

# all_df["KitchenQual-2"] = all_df["KitchenQual"]**2
# all_df["KitchenQual-3"] = all_df["KitchenQual"]**3
# all_df["KitchenQual-Sq"] = np.sqrt(all_df["KitchenQual"])

# all_df["GarageScore-2"] = all_df["GarageScore"]**2
# all_df["GarageScore-3"] = all_df["GarageScore"]**3
# all_df["GarageScore-Sq"] = np.sqrt(all_df["GarageScore"])

# #all_df["GarageArea-2"] = all_df["GarageArea"] ** 2
# #all_df["GarageArea-3"] = all_df["GarageArea"] ** 3
# #all_df["GarageArea-Sq"] = np.sqrt(all_df["GarageArea"])

# #all_df["LotArea-2"] = all_df["LotArea"] ** 2
# #all_df["LotArea-3"] = all_df["LotArea"] ** 3
# #all_df["LotArea-Sq"] = np.sqrt(all_df["LotArea"])
# #all_df["LotFrontage-2"] = all_df["LotFrontage"] ** 2
# #all_df["LotFrontage-3"] = all_df["LotFrontage"] ** 3
# #all_df["LotFrontage-Sq"] = np.sqrt(all_df["LotFrontage"])
# #all_df["OverallGrade-2"] = all_df["OverallGrade"] ** 2
# #all_df["OverallGrade-3"] = all_df["OverallGrade"] ** 3
# #all_df["OverallGrade-Sq"] = np.sqrt(all_df["OverallGrade"])
