# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:04:28 2024

@author: Mason
"""
from data_pipeline import Preprocessing


data_class = Preprocessing(valid_data_path = '../data/test.csv',
                           train_data_path = '../data/train.csv',
                           target_col = 'sii',
                           categorical_values = ['Spring', 'Summer', 'Fall', 'Winter'],
                           split_percentage = 0.85)

train, valid = data_class.data_cleaning(data_class.train, data_class.valid)
tensor_data = data_class.tensor_conversion(train, valid, data_class.targets)
train_data, test_data, train_target, test_target, valid_data = tensor_data