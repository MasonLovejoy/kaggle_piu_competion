# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:03:25 2024

@author: Mason
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
import torch

class Preprocessing:
    """
    This class will convert our csv's into a useful tensors
    """
    def __init__(self, 
                 valid_data_path: str, 
                 train_data_path: str, 
                 target_col: str,
                 categorical_values: list[str],
                 split_percentage: float,
                 missing_data_amt: float):
        
        self.split_perc = split_percentage
        self.cat_vals = categorical_values
        self.valid = pd.read_csv(valid_data_path)
        self.train = pd.read_csv(train_data_path)
        self.train = self.train.dropna(axis=0, subset='sii')
        self.targets = self.train[target_col]
        self.missing_cap = missing_data_amt
    
    def data_cleaning(self, train, valid):
        # Removing the columns that do not exist in the validation set
        for col in train:
            if col not in valid.columns:
                train = train.drop(columns = col) 
        
        split_val = len(train.index)
                
        # Combining the datasets in order to help with normalizing and
        # data imputation
        data = pd.concat([train, valid], ignore_index=True, axis=0)
        ids = data['id']
        
        data = data.drop(columns = 'id')
        null_df = data.isnull().sum()
        for col in null_df.index:
            if (null_df[col]/len(data)) > self.missing_cap:
                data = data.drop(columns = col)
        
        # Replacing Categorical Values
        data = data.replace(self.cat_vals, list(range(len(self.cat_vals))))
        
        # Randomly adding new values based on the distribution of given values
        for feature in data.columns:
            data['rand_' + feature] = data[feature]
            data = self.random_imputation(data, feature)
        
        # Creating a list of cols that are missing values
        null_df = data.isnull().sum()
        missing_columns = []
        for index in null_df.index:
            if null_df[index] != 0:
                missing_columns.append(index)
                
        # Creating a new dataframe to add values to
        lr_data = pd.DataFrame(columns = ['lr_' + name for name in data.columns])
        
        # Creates a Linear Regression model for each random variable the uses
        # it to add missing values
        for feature in missing_columns:
            lr_data['lr_' + feature] = data['rand_' + feature]
            parameters = list(set(data.columns) - set(missing_columns) - {'rand_' + feature})
        
            model = linear_model.LinearRegression()
            model.fit(X = data[parameters], y = data['rand_' + feature])
        
            lr_data.loc[data[feature].isnull(), 'lr_' + feature] = model.predict(data[parameters])[data[feature].isnull()]   
        
        # Removing columns with missing data
        lr_null_df = lr_data.isnull().sum()
        for feature in lr_null_df.index:
            if lr_null_df[feature] != 0 :
                lr_data = lr_data.drop(columns=feature)
                
        # Normalizing Data
        lr_data = (lr_data-lr_data.min()) / (lr_data.max() - lr_data.min())
                
        self.valid_ids = ids.iloc[split_val:]
        train = lr_data.iloc[:split_val, :]
        valid = lr_data.iloc[split_val:, :]
    
        assert len(train.index) == len(self.train.index)
        return [train, valid]
        
    def tensor_conversion(self, training_data, validation_data, targets):
        train_array = training_data.to_numpy()
        target_array = targets.to_numpy()
        valid_array = validation_data.to_numpy()

        train_tensor = torch.tensor(train_array)
        target_tensor = torch.tensor(target_array)
        valid_data = torch.tensor(valid_array)
        
        split_val = int(round(self.split_perc*len(self.train.index), -1))
        
        train_data = train_tensor[:split_val]
        test_data = train_tensor[split_val:]
        
        train_target = target_tensor[:split_val]
        test_target = target_tensor[split_val:]
        
        return [train_data, test_data, train_target, test_target, valid_data]

                
        
    def random_imputation(self, df, feature):
        number_missing = df[feature].isnull().sum()
        observed_values = df.loc[df[feature].notnull(), feature]
        df.loc[df[feature].isnull(), 'rand_' + feature] = np.random.choice(observed_values, number_missing, replace = True)
        return df