# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:44:44 2024

@author: Mason
"""
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Training():
    def __init__(self,
                 tensor_data: list[torch.tensor],
                 use_cuda: bool,
                 model,
                 number_epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 ids):
        
        # tensor_data is comming from the preprocessing class
        self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.valid_data = tensor_data
        self.model = model(first_dim = self.Xtrain.size()[1])

        # loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
         
        self.n_epochs = number_epochs
        self.batch_size = batch_size
        self.batches_per_epoch = len(self.Xtrain) // self.batch_size
        
        if use_cuda:
            self.Xtrain = self.Xtrain.type(torch.float64).cuda()
            self.Xtest = self.Xtest.type(torch.float64).cuda()
            self.ytrain = self.ytrain.type(torch.float64).cuda()
            self.ytest = self.ytest.type(torch.float64).cuda()
            self.valid_data = self.valid_data.type(torch.float64).cuda()
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()
            
        loss_list = []
            
        for epoch in range(self.n_epochs):
            epoch_loss_list = []
            for i in range(self.batches_per_epoch):
                start = i * batch_size
                # take a batch
                Xbatch = self.Xtrain[start:start+batch_size].cuda()
                ybatch = self.ytrain[start:start+batch_size]
                
                if use_cuda:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                
                # forward pass
                y_pred = self.model(Xbatch)
            
                ybatch = ybatch.type(torch.LongTensor)
                Xbatch = Xbatch.type(torch.LongTensor)
                
                if use_cuda:
                    ybatch = ybatch.cuda()
                    Xbatch = Xbatch.cuda()
            
                loss = self.loss_fn(y_pred, ybatch)
                loss_cpu = loss.cpu()
                epoch_loss_list.append(loss_cpu.detach().numpy())
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()
            
            avg_epoch_loss = np.mean(epoch_loss_list)
            loss_list.append(np.mean(avg_epoch_loss))
            print('Epoch Number: ' + str(epoch))
            print('Current Loss: ' + str(avg_epoch_loss))
            
        # evaluate trained model with test set
        with torch.no_grad():
            y_pred_final = self.model(self.Xtest)

        y_pred_final = torch.argmax(y_pred_final, dim=1)
        y_pred_final = y_pred_final.type(torch.float64)

        preds = []
        reals = []

        for i in y_pred_final:
            preds.append(i.item())
            
        for i in self.ytest:
            reals.append(i.item())
            
        error_list = []

        for i in range(0, len(preds)):
            error_list.append(np.sqrt((preds[i]-reals[i])**2))
            
        error_tot = float(np.average(error_list))

        print("Average Error: " +str(error_tot))
        print("Correct: " + str(error_list.count(0.0)) +"/" +str(len(error_list)))
        
        with torch.no_grad():
            self.valid_preds = self.model(self.valid_data)
        
        self.valid_preds = torch.argmax(self.valid_preds, dim=1)
        self.valid_preds = self.valid_preds.type(torch.float64)
        
        pred_list = []
        for i in self.valid_preds:
            pred_list.append(i.item())
        
        valid_preds_df = pd.DataFrame({'ids': ids,
                                       'preds': pred_list})
        self.valid_preds = valid_preds_df
        x = np.array(range(0, self.n_epochs))
        y = np.array(loss_list)
        plt.plot(x, y)
        plt.show()
        
