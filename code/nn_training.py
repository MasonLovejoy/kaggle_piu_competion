# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:37:38 2024

@author: Mason
"""
from data_pipeline import Preprocessing
from training_pipeline import Training
from nn_model import Model

data_class = Preprocessing(valid_data_path = '../data/test.csv',
                           train_data_path = '../data/train.csv',
                           target_col = 'sii',
                           categorical_values = ['Spring', 'Summer', 'Fall', 'Winter'],
                           split_percentage = 0.85,
                           missing_data_amt = 0.60)

train, valid = data_class.data_cleaning(data_class.train, data_class.valid)
tensors = data_class.tensor_conversion(train, valid, data_class.targets)

training_class = Training(tensor_data = tensors,
                          use_cuda = True,
                          model = Model,
                          number_epochs = 600,
                          batch_size = 25,
                          learning_rate = 0.003,
                          ids = data_class.valid_ids)

preds = training_class.valid_preds
