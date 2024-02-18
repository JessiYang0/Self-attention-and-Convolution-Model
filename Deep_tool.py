# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:23:14 2020

@author: Userr
"""

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split



def create_datasets(Group, target_group, test_size=0.2):

    X_train, X_valid, y_train, y_valid = train_test_split(Group, target_group, test_size=0.2)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    
    return [train_ds, valid_ds, target_group]



def create_loaders(train_ds, valid_ds, bs=32, jobs=0):
    
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    
    return train_dl, valid_dl





class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]



def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler



def LR_sche(optimizer,learning_rate,epoch):
    return CyclicLR(optimizer, cosine(t_max=epoch * 2, eta_min=learning_rate/100))
    


def MSE(forecast, prediction):
        
        y_hat  = forecast.reshape(-1)    
        y_pred = prediction.reshape(-1)    
        mse = torch.mean((y_hat - y_pred)**2) 
   
        return mse
  
class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers,output_size):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        
        self.fc =   nn.Sequential(
            nn.Dropout(),
            nn.Linear(17500, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8192, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_size),
        )
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        out = torch.flatten(out,1)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden



class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        self.fc =   nn.Sequential(
            nn.Dropout(),
            nn.Linear(17500, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8192, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim),
        )
        
        self.batch_size = None
        self.hidden = None
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
    
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


class SimpleCNN_model(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128       , 256, kernel_size=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256      , 512, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(512     , 512, kernel_size=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=3, stride=2),
        )
        
        self.fullconnect = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )
    
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fullconnect(x)
        
        return x
    
   
class WaveletCNN_model(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64       , 128, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128      , 256, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(256     , 512, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(512     , 512, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        self.fullconnect = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )
    
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fullconnect(x)
        
        return x



    
def WAPE(true_value, prediction):
    
    y_true  = true_value.reshape(-1)    
    y_pred  = prediction.reshape(-1)    
    
    #wap = (np.sum(abs(y_true-y_pred))/np.sum(y_true))/len(y_true)
    wape     = (np.sum(((y_true - y_pred)/np.mean(y_true))))/len(y_true)
   
    return abs(wape) 




def MAE(true_value,prediction):
    y_true  = true_value.reshape(-1)    
    y_pred  = prediction.reshape(-1)    
    mae = (np.sum(abs(y_true-y_pred)))/len(y_true)
    
    return mae
    
    
def MSE2(true_value,prediction):
    y_true  = true_value.reshape(-1)    
    y_pred  = prediction.reshape(-1)    
    mse = np.sum((y_true-y_pred)**2)/len(y_true)
    
    return mse


def score(true_value,prediction):
    print('MAE score is: ' , MAE(true_value,prediction))
    print('MSE score is: ' , MSE2(true_value,prediction))
    print('RMSE score is: ', MSE2(true_value,prediction)**0.5)
    print('WAPE score is: ', WAPE(true_value,prediction))
    
    
