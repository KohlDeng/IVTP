# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:45:47 2022

@author: kohldyh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/sample"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

from hjdriveset import HjdriveSet

#%%
#model define

class IVLSTM(nn.Module):
    def __init__(self,
                in_feature=6,
                in_steps=50,
                out_feature=6,
                out_steps=100,
                num_layers=2,
                dropout=0.2,
                device='cpu'):
        super().__init__()
        self.in_feature = in_feature
        self.in_steps = in_steps
        self.out_feature = out_feature
        self.out_steps = out_steps
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_feature = 64
        self.dev=device
        
        self.lstm = nn.LSTM(self.in_feature,self.hidden_feature,num_layers=num_layers)
        
        self.linear1 = nn.Linear(self.in_steps*self.hidden_feature,256*self.hidden_feature)
        self.linear2 = nn.Linear(256*self.hidden_feature,self.out_steps*self.out_feature)
        # self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        # x = F.relu(self.linear(x))
        batch_size = x.shape[0]
        x = torch.transpose(x, 0, 1)
        h0 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev)
        c0 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev)
        hidden,(hn,cn) = self.lstm(x,(h0,c0))
        y1 = torch.transpose(hidden,0,1)
        y = torch.reshape(y1, (batch_size,-1))
        y = F.relu(self.linear1(y))
        y = self.linear2(y)
        
        return y.view(batch_size,self.out_steps,self.out_feature)
        
if __name__=="__main__":    
    dev = 'cpu'
    in_feature=6
    in_steps=50
    out_feature=6
    out_steps=100
    num_layers=2
                                    
    model = IVLSTM(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev)
    dataset = HjdriveSet(sample_dir)
        
    data_raw,label = dataset.__getitem__(2)
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)
       
    raws = torch.zeros(1024,50,6)
    raws[0] = raw
    
    output = model(raws)
    print(model)
    print(output.shape)
    
    

    from thop import profile
    flops,params = profile(model,inputs=(raws,)) #flops和参数量
    print("the floating point operation per second is:{}M".format(flops/1024**2))
    print("the number of params is {}M".format(params/1024**2))






