# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:08:15 2024

@author: Mason
"""
import numpy as np
from sklearn import linear_model
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
imp = pd.read_csv('data/imputed_train.csv')

missing_cols = []
for col in train:
    if col not in test.columns:
        if col != 'sii':
            missing_cols.append(col)
        
train = train.drop(columns = missing_cols)
train = train.replace(['Spring', 'Summer', 'Fall', 'Winter'], [0,1,2,3])
nul_train = train.isnull().sum()

def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

for feature in train.columns:
    if feature != 'sii':
        train[feature + '_imp'] = train[feature]
        train = random_imputation(train, feature)

deter_data = pd.DataFrame(columns = ["Det" + name for name in train.columns])

missing_columns = []
for index in nul_train.index:
    if nul_train[index] != 0:
        missing_columns.append(index)


for feature in missing_columns:
    if feature == 'sii':
        deter_data['sii'] = train['sii']
    else:
        deter_data['Det' + feature] = train[feature + '_imp']
        parameters = list(set(train.columns) - set(missing_columns + ['id'] +['id_imp'] + ['sii']) - {feature + '_imp'})
    
        model = linear_model.LinearRegression()
        check_x_dtypes = train[parameters].dtypes
        model.fit(X = train[parameters], y = train[feature + '_imp'])
    
        deter_data.loc[train[feature].isnull(), 'Det' + feature] = model.predict(train[parameters])[train[feature].isnull()]

isnull = deter_data.isnull().sum()

for feature in isnull.index:
    if isnull[feature] != 0 :
        if feature != 'sii':
            deter_data = deter_data.drop(columns=feature)

#deter_data model
deter_data = deter_data.dropna(subset='sii')
deter_data_train = deter_data[:2300]
deter_data_test = deter_data[2300:]

dy_train = deter_data_train['sii']
dy_test = deter_data_test['sii']
dx_train = deter_data_train.drop(columns='sii')
dx_test = deter_data_test.drop(columns='sii')

model_deter = linear_model.LinearRegression()

model_deter.fit(X = dx_train, y = dy_train)

prediction = model_deter.predict(dx_test)
predition = prediction.round()

for i in range(0, len(prediction)):
    prediction[i] += 1

dy_test = dy_test.to_numpy()

error_array = dy_test-prediction
error = np.average(error_array)

#imputed_avg model
imp = imp.dropna(subset='sii')
imp_train = imp[:2300]
imp_test = imp[2300:]

imp_dy_train = imp_train['sii']
imp_dy_test = imp_test['sii']
imp_dx_train = imp_train.drop(columns='sii')
imp_dx_test = imp_test.drop(columns='sii')

model_imp = linear_model.LinearRegression()

model_imp.fit(X = imp_dx_train, y = imp_dy_train)
imp_pred = model_imp.predict(imp_dx_test)
imp_pred = prediction.round()

for i in range(0, len(imp_pred)):
    imp_pred[i] += 1
    
imp_dy_test = imp_dy_test.to_numpy()

imp_error = imp_dy_test - imp_pred
imp_rms = np.average(imp_error)

print('---------------------------------------------')
print('Imputed Data using Averages: ', imp_rms)
print('Imputed Data using Linear Regression: ', error)
print('---------------------------------------------')

