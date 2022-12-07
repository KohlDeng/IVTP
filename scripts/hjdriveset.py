# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 2022

@author: kohldyh
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset,random_split
from sklearn.preprocessing import MinMaxScaler


sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/hjdrive_30000"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"



class HjdriveSet(Dataset):
    
    def __init__(self,raw_lbl_dir,ifscaler=False,scaler_file=None):
        self.raw_lbl_dir=raw_lbl_dir
        # self.transform = transform
        self.list_data = os.listdir(self.raw_lbl_dir)
        self.ifscaler = ifscaler
        self.scaler_file = scaler_file
        
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self,idx):
        data_raw_name = self.list_data[idx]
        data_raw_dir = os.path.join(self.raw_lbl_dir,data_raw_name)
        
        csvdata = pd.read_csv(data_raw_dir, header=None)
        csvdata.columns =["mcutime", "latency", "num_obstacles", "id",
                          "px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration",
                          "heading_rel","cipv_type"]
        label=["px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration"]
        alldata=csvdata[label]
        
        x = alldata.iloc[:50]      #get 50 rows from table
        y = alldata.iloc[49:]      #get 100 rows from table

        data_raw = np.array(x,dtype=np.float32)
        label = np.array(y,dtype=np.float32)
        
        if self.ifscaler:
            path = os.path.join(prj_dir,self.scaler_file)
            if os.path.isfile(path): 
                MinMaxInfo = np.load(path)
                minInfo = MinMaxInfo[1,:]
                scalerInfo = MinMaxInfo[3,:]*100
                
                data_raw = (data_raw-minInfo)*scalerInfo
                label = (label-minInfo)*scalerInfo
            else:
                print("no scaler npy file")
                exit()
           
        data = (data_raw,label)
        return data

if __name__=="__main__":     
#     dataset = HjdriveSet(sample_dir)
#     print(dataset.__len__())
#     num = dataset.__len__()
#     steps = 150    
# #%%-------------------------------------------------------------------    
#     alldata = np.zeros((num*steps,6))
#     for idx in range(num):
#         xx,yy = dataset.__getitem__(idx)
#         data = np.concatenate([xx,yy],axis=0)
#         alldata[steps*idx:steps*(idx+1),:]=data
#     print(alldata.shape)
#     scaler = MinMaxScaler()
#     scaler.fit_transform(alldata)
#     minInfo = scaler.data_max_
#     maxInfo = scaler.data_min_
#     rangeInfo = scaler.data_range_
#     scalerInfo = 1/rangeInfo
#     MinMaxInfo = np.zeros((4,6),dtype=np.float32)
#     MinMaxInfo[0,:]=minInfo
#     MinMaxInfo[1,:]=maxInfo
#     MinMaxInfo[2,:]=rangeInfo
#     MinMaxInfo[3,:]=scalerInfo

#     print(MinMaxInfo)
#     np.save('config/hjdrive_30000_mminfo.npy',MinMaxInfo)
#%%-----------------------------------------------------------------    
    dataset2 = HjdriveSet(sample_dir,ifscaler=True,scaler_file='config/hjdrive_30000_mminfo.npy')
    xx,yy = dataset2.__getitem__(2)
    print(xx.shape,yy.shape)

    full_size = len(dataset2)  
    print(full_size)
    train_size = int(0.8*full_size)
    val_size = full_size - train_size
    torch.manual_seed(0)
    trainset,valset = random_split(dataset2, [train_size,val_size])
    print(len(valset))
    print(trainset.__len__())

