# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:45:47 2022

@author: kohldyh
"""


import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init
from torchsummary import summary
from thop import profile


import os
import numpy as np

from hjdriveset import HjdriveSet

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/sample"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"


#%%
#model define

class MLPmodel(nn.Module):
    def __init__(self,num_inputs,num_hiddens,num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        
        self.linear1=nn.Linear(num_inputs,num_hiddens)
        self.relu = nn.ReLU()
        self.linear2=nn.Linear(num_hiddens,num_outputs)
    def forward(self,x):
        # assert(x.shape[1]*x.shape[2]==self.num_inputs,'inputs dim error')
        a0 = x.view(x.shape[0],-1)
        z1 = self.linear1(a0)
        a1 = self.relu(z1)
        y = self.linear2(a1)
        
        return y.view(y.shape[0],self.num_outputs//6,6)

if __name__=="__main__":
    dev = 'cuda:0'
    num_inputs = 50*6
    num_hiddens = 256*6
    num_outputs = 100*6
    model = MLPmodel(num_inputs, num_hiddens, num_outputs).to(dev)
    print(model)
    dataset = HjdriveSet(sample_dir)
    
    data_raw,label = dataset.__getitem__(2)
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)
    
    
    inputs = torch.zeros(1,50,6).to(dev)
    inputs[0] = raw
    
    # inputs = raw_tensor.unsequence(dim=0)
    # labels = label.view(1,*label.size())
    
    output = model(inputs)
    
    print(output.size())
    # print(raw)
    # tmp = inputs.view(inputs.shape[0],-1)
    # print(tmp.view(inputs.shape[0],50,6))
    
    # summary(model,input_size=inputs.shape)
    
    flops,params = profile(model,inputs=(inputs,)) #flops和参数量
    print("the floating point operation per second is:{}M".format(flops/1024**2))
    print("the number of params is {}M".format(params/1024**2))
    
    train_size = int(0.8 * len(full_dataset))
    
    
    
    
    
   