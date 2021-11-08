import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from torch import optim
from torch.autograd import Variable

import time
import math
import numpy as np

import os
import sys
import random
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 指定训练的GPU


def random_mini_batches(XE, R1E, mini_batch_size = 10, seed = 42):          
    m = XE.shape[0]                  
    mini_batches = []
    num_complete_minibatches = math.floor(m/mini_batch_size)
    
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_X1R = R1E[k * mini_batch_size : (k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_XE, mini_batch_X1R)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = XE[Lower : Lower + Upper, :]
        mini_batch_X1R = R1E[Lower : Lower + Upper]
        
        mini_batch = (mini_batch_XE, mini_batch_X1R)
        mini_batches.append(mini_batch)
    
    return mini_batches
    

class SimpleNet(nn.Module):
    def __init__(self):

        super(SimpleNet,self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,3)
        )
            
    def forward(self, x):
        x = self.fc(x)
        return x

############################ normalization #############################
#  input: X-(nosample, channels, height, width)
#  output: normalized X to (0,1)
# 对每一维针对所有样本做归一化
#######################################################################
def data_normalize(X):
    #X = np.expand_dims(X, 1)
    s = X.shape
    y = np.zeros(s)
    for cc in range(s[1]):
        MinValue = min(X[:,cc])
        MaxValue = max(X[:,cc])
        if MinValue==MaxValue:
            #print('MinValue==MaxValue')
            y[:,cc] = X[:,cc]     # keep the same
        else:
            y[:,cc] = (X[:,cc]-MinValue)/(MaxValue-MinValue)*2-1
    return y



def train(X, Y, batch_size, LR, flag, sample_index):
    noval = 1000

    train_index = list(range(X.shape[0]))               # 将序列a中的元素顺序打乱
    #random.shuffle(train_index)

    val_index = list(range(noval))
    #random.shuffle(val_index)

    print(X.shape)

    X_train = X[train_index,0:12]
    X_valid = X[val_index,0:12]

    Y_train = Y[train_index,0:3]
    Y_valid = Y[val_index,0:3]

    #X_train = data_normalize(X_train)
    #X_valid = data_normalize(X_valid)

    X_train = torch.from_numpy(X_train)
    X_train = X_train.float()
    Y_train = torch.tensor(Y_train)
    Y_train = Y_train.float()

    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.float()
    Y_valid = torch.tensor(Y_valid)
    Y_valid = Y_valid.float()


    model = SimpleNet()

    input_size = X_train.shape[0]

    loss_val = []

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

    loss_func = nn.MSELoss()

    epoch = 300
    for it in range(epoch):
        epoch_cost = 0
        
        num_minibatches = int(input_size / batch_size)
        
        minibatches = random_mini_batches(X_train, Y_train, batch_size)
        
        for minibatch in minibatches:
            XE, YE = minibatch 
            # XE, YE = XE.cuda(), YE.cuda()
            
            Yhat = model(XE)
            
            loss = loss_func(Yhat, YE)    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_cost = epoch_cost + (loss / num_minibatches)

        #exp_lr_scheduler.step()
        
        with torch.no_grad():
            Yhat1D = model(X_valid)#.cuda()
            loss_validation = loss_func(Yhat1D, Y_valid)       
            loss_val.append(loss_validation)
            
        print('Iter-{} || Training loss: {:.4} || validation loss: {:.4}'.format(it, epoch_cost, loss_validation.data.item()))
    
    # 每种情况一个模型  
    torch.save(model.state_dict(), './model_{}_{}.pt'.format(flag,sample_index))

def test(X_test, flag, sample_index):
    #X_test = data_normalize(X_test)
    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()

    current_path = os.path.dirname(__file__)
    # print(f"current_path: {current_path}")
    m_state_dict = torch.load(os.path.join(current_path, './model_{}_{}.pt'.format(flag,sample_index)))
    new_model = SimpleNet()#.cuda()
    
    new_model.load_state_dict(m_state_dict)
 
    Yhat1D  = new_model(X_test)#.cuda())

    Yhat1D = Yhat1D.cpu().detach().numpy() 

    return Yhat1D
