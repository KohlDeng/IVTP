# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:39:30 2022

@author: DELL
"""
import os
from math import sqrt

import torch
import numpy as np
import matplotlib.pyplot as plt

from hjdriveset import HjdriveSet
from mlp import MLPmodel
from lstm import IVLSTM

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/sample"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

dev = 'cpu'
# 模型加载和数据集
# IVLSTM
# in_feature=6
# in_steps=50
# out_feature=6
# out_steps=100
# num_layers=1                            
# model = IVLSTM(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev)
# weight_file = os.path.join(prj_dir,'checkpoint/lstm_weight_epoch_200.pth.tar')
#MLPModel
num_inputs = 50*6
num_hiddens = 256*6
num_outputs = 100*6
model = MLPmodel(num_inputs, num_hiddens, num_outputs).to(dev)
weight_file = os.path.join(prj_dir,'checkpoint/mlp_weight_epoch_1000.pth.tar')

if os.path.isfile(weight_file): 
    print("Loading checkpoint '{}'".format(weight_file))
    checkpoint = torch.load(weight_file)
    epoch = checkpoint['epoch']
    print("epoch of weight file is:",epoch)
    
    list_loss = checkpoint['list_loss']
    print('last loss is:',list_loss[-1])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    # print(checkpoint[''])
else:
    print("no weights file")
    exit()
        
        
ifscaler = True
scaler_file = 'config/hjdrive_30000_mminfo.npy'
testset = HjdriveSet(sample_dir,ifscaler=ifscaler,scaler_file=scaler_file)

num = testset.__len__()
image_num = 9

ade=[]
fde=[]
ade_x=[]
fde_x=[]
ade_y=[]
fde_y=[]
print('start test....')
for i in range(num):
    data_raw,label = testset.__getitem__(i)
    
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)
    raws = torch.zeros(1,50,6)
    raws[0] = raw
    outputs = model(raws)  
    predict = outputs[0].detach().numpy()
    
    if(ifscaler):
        path = os.path.join(prj_dir,scaler_file)
        if os.path.isfile(path): 
            MinMaxInfo = np.load(path)
            minInfo = MinMaxInfo[1,:]
            scalerInfo = MinMaxInfo[3,:]*100
            
            predict = predict/scalerInfo+minInfo
            data_raw = data_raw/scalerInfo+minInfo
            label = label/scalerInfo+minInfo    
    
    x1 = data_raw[:,0]
    y1 = data_raw[:,1]
    x2 = label[:,0]
    y2 = label[:,1]
    x3 = predict[:,0]
    y3 = predict[:,1]
    
    err_x = abs(x2-x3)
    err_y = abs(y2-y3)
    err = np.square(err_x)+np.square(err_y)
    err = np.sqrt(err)
    # err = np.sqrt(err_x)
    
    ade_x.append(np.mean(err_x))
    ade_y.append(np.mean(err_y))
    ade.append(np.mean(err))
    
    k=i+23;
    p = round(num/image_num);
    if(k%p==0):
        plt.figure(figsize=(10,6))
        plt.plot(x1,y1, '*',label='Past Data')
        plt.plot(x2,y2, label='labeled Data')
        plt.plot(x3,y3,'--', label='predicted Data')
        # plt.axis([-1,1,-1,1])
        plt.axis('equal')
        plt.legend()
                 

    
print('test size is:',num)    
print('ade_x mean is:',np.mean(ade_x))
print('ade_y mean is:',np.mean(ade_y))
print('ade mean is:',np.mean(ade))
print('ade max is:',np.max(ade))

plt.show()     

    
    
