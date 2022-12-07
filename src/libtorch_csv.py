#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Nov 2022 

@author: kohldyh
"""
import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev = "cpu"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"


filepath = 'dataset/sample/Data103_187.csv'
scaler_file = 'config/hjdrive_30000_mminfo.npy'

csvdata = pd.read_csv(filepath, header=None)

csvdata.columns =["mcutimes", "latency", "num_obstacles", "id",
                  "px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration",
                  "heading_rel","cipv_type"]
label=["px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration"]

alldata = csvdata[label]

x = alldata.iloc[:50]           #get 50 rows from table
y = alldata.iloc[49:]

data_raw = np.array(x,dtype=np.float32)
label = np.array(y,dtype=np.float32)

scaler_path = os.path.join(prj_dir,scaler_file)
MinMaxInfo = np.load(scaler_path)
minInfo = MinMaxInfo[1,:]
scalerInfo = MinMaxInfo[3,:]*100
data_raw = (data_raw-minInfo)*scalerInfo
label = (label-minInfo)*scalerInfo


#print(device)
pt_file = "src/mlp_20221118.pt"
pt_path = os.path.join(prj_dir,pt_file)
model = torch.jit.load(pt_path).to(dev)

raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)

# print(raw)

inputs = torch.zeros(1,50,6)
inputs[0] = raw

outputs = model(inputs)

# print(outputs.size())

predict = outputs[0].detach().numpy()
print(predict)
    
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
plt.plot(x3,y3, label='predicted Data')
# plt.axis([-1,1,-1,1])
plt.axis('equal')
plt.legend()
plt.show()             


# np.savetxt('config/hjdrive_30000_mminfo.txt', MinMaxInfo, fmt='%0.18f')
