# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:45:47 2022

@author: kohldyh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/hjdrive_30000"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

from hjdriveset import HjdriveSet

class Seq2Seq(nn.Module):
    def __init__(self,
                in_feature=6,
                in_steps=50,
                out_feature=6,
                out_steps=100,
                num_layers=2,
                dropout=0.2,
                device = 'cpu'):
        super().__init__()
        self.in_feature = in_feature
        self.in_steps = in_steps
        self.out_feature = out_feature
        self.out_steps = out_steps
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_feature = 32
        self.dev=device
        
        self.lstm1 = nn.LSTM(self.in_feature,self.hidden_feature,num_layers=num_layers)
        self.lstm2 = nn.LSTM(self.hidden_feature,self.hidden_feature,num_layers=num_layers)
        #linear input size is batch_size,out_steps*hidden_feature
        
        self.linear1 = nn.Linear(self.out_steps*self.hidden_feature,256*self.hidden_feature)
        self.linear2 = nn.Linear(256*self.hidden_feature,128*self.hidden_feature)
        self.linear3 = nn.Linear(128*self.hidden_feature,64*self.hidden_feature)
        self.linear4 = nn.Linear(64*self.hidden_feature,self.out_steps*self.out_feature)
        
    def forward(self,x):
        batch_size = x.shape[0]
        #encoding
        x = torch.transpose(x, 0, 1)
        h01 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev)
        c01 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev)        
        hidden,(hn1,cn1) = self.lstm1(x,(h01,c01))
        z = hn1.expand(self.out_steps,batch_size,self.hidden_feature)
        #decoding
        h02 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev)
        c02 = torch.zeros(self.num_layers,batch_size,self.hidden_feature,device=self.dev) 
        z,(hn2,cn2) = self.lstm2(z,(h02,c02))
        z = torch.transpose(z,0,1)
        #header network
        y = torch.reshape(z, (batch_size,-1))
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))
        y = F.relu(self.linear3(y))
        y = self.linear4(y)

        return y.view(batch_size,self.out_steps,self.out_feature)

            
if __name__=="__main__":    
    dev = 'cpu'
    in_feature=6
    in_steps=50
    out_feature=6
    out_steps=100
    num_layers=1
    
    model = Seq2Seq(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev)
    dataset = HjdriveSet(sample_dir)    
    
    data_raw,label = dataset.__getitem__(2)
    raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)
   
    raws = torch.zeros(5,50,6)
    raws[0] = raw
    
    output = model(raws)
    print(model)
    print(output.shape)
    
    from thop import profile
    flops,params = profile(model,inputs=(raws,)) #flops和参数量
    print("flops is:{}M".format(flops/1024**2))
    print("params is {}M".format(params/1024**2))


