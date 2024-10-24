# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:37:38 2024

@author: Mason
"""
import numpy as np
import torch
import torch.optim as optim
from data_pipeline import Preprocessing
from nn_model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_class = Preprocessing(valid_data_path = '../data/test.csv',
                           train_data_path = '../data/train.csv',
                           target_col = 'sii',
                           categorical_values = ['Spring', 'Summer', 'Fall', 'Winter'],
                           split_percentage = 0.85)

train, valid = data_class.data_cleaning(data_class.train, data_class.valid)
tensor_data = data_class.tensor_conversion(train, valid, data_class.targets)
Xtrain, Xtest, ytrain, ytest, valid_data = tensor_data

Xtrain = Xtrain.type(torch.float64).cuda()
Xtest = Xtest.type(torch.float64).cuda()
ytrain = ytrain.type(torch.float64).cuda()
ytest = ytest.type(torch.float64).cuda()

model = Model().cuda()

# loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
n_epochs = 500
    # number of epochs to run
batch_size = 25  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size
 
loss_list = []
 
for epoch in range(n_epochs):
    epoch_loss_list = []
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size].cuda()
        ybatch = ytrain[start:start+batch_size].cuda()
        
        # forward pass
        y_pred = model(Xbatch).cuda()
    
        ybatch = ybatch.type(torch.LongTensor).cuda()
        Xbatch = Xbatch.type(torch.LongTensor).cuda()
    
        loss = loss_fn(y_pred, ybatch)
        loss_cpu = loss.cpu()
        epoch_loss_list.append(loss_cpu.detach().numpy())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    
    avg_epoch_loss = np.mean(epoch_loss_list)
    loss_list.append(np.mean(avg_epoch_loss))
    print('Epoch Number: ' + str(epoch))
    print('Current Loss: ' + str(avg_epoch_loss))

# evaluate trained model with test set
with torch.no_grad():
    y_pred_final = model(Xtest)

y_pred_final = torch.argmax(y_pred_final, dim=1)
y_pred_final = y_pred_final.type(torch.float64)

preds = []
reals = []

for i in y_pred_final:
    preds.append(i.item())
    
for i in ytest:
    reals.append(i.item())
    
error_list = []

for i in range(0, len(preds)):
    error_list.append(np.sqrt((preds[i]-reals[i])**2))
    
error_tot = float(np.average(error_list))

print("Average Error: " +str(error_tot))
print("Correct: " + str(error_list.count(0.0)) +"/" +str(len(error_list)))
