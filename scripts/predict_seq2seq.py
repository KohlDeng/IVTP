# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:45:47 2022

@author: kohldyh
"""

import os
from hjdriveset import HjdriveSet
from seq2seq import Seq2Seq

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
    
    model = Seq2Seq(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev)
    weight_file = os.path.join(prj_dir,'checkpoint/seq_weight_epoch_400.pth.tar')

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
    dataset = HjdriveSet(sample_dir,ifscaler=ifscaler,scaler_file=scaler_file)

    data_raw,label = dataset.__getitem__(122)
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)

    inputs = torch.zeros(1,50,6)
    inputs[0] = raw
    
    outputs = model(inputs)
    
    print(outputs.size())
    
    predict = outputs[0].detach().numpy()
    print(predict)
    
    if(ifscaler):
        path = os.path.join(prj_dir,scaler_file)
        if os.path.isfile(path): 
            MinMaxInfo = np.load(path)
            # print(MinMaxInfo)
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
    traced_script_module.save("src/seq2seq_20221118.pt")     
    
    
    