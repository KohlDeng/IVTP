# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:45:47 2022

@author: kohldyh
"""


import os
from hjdriveset import HjdriveSet
from lstm import IVLSTM

import torch
import matplotlib.pyplot as plt
import numpy as np

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/sample"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

    
    
if __name__=="__main__":
    dev = 'cpu'
    in_feature=6
    in_steps=50
    out_feature=6
    out_steps=100
    num_layers=1
    model = IVLSTM(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev)
    
    weight_file = os.path.join(prj_dir,'checkpoint/lstm_weight_epoch_240.pth.tar')
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
    
    
    #for test
    # dataset = HjdriveSet(sample_dir)
    ifscaler = True
    scaler_file = 'config/sample_mminfo.npy'
    dataset = HjdriveSet(sample_dir,ifscaler=ifscaler,scaler_file='config/sample_mminfo.npy')
    
    
    data_raw,label = dataset.__getitem__(1036)
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)

    raws = torch.zeros(1,50,6)
    raws[0] = raw
    outputs = model(raws)
    print(outputs)    
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

               
    #%% plot
    x1 = data_raw[:,0]
    y1 = data_raw[:,1]
    x2 = label[:,0]
    y2 = label[:,1]
    x3 = predict[:,0]
    y3 = predict[:,1]
    
    plt.figure(figsize=(10,6))
    plt.plot(x1,y1, '*',label='Past Data')
    plt.plot(x2,y2, label='labeled Data')
    plt.plot(x3,y3,'.', label='predicted Data')
    # plt.axis([-1,1,-1,1])
    plt.axis('equal')
    plt.legend()
    plt.show()  
    
    #%% 生成C++可调用模型
    example_input = torch.ones(1, in_steps, in_feature)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("src/lstm_20221118.pt")  
    