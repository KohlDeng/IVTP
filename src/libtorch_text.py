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
dev = "cpu"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

data_raw = np.loadtxt('src/data.txt')

#print(device)
pt_file = "src/mlp_20221118.pt"
pt_path = os.path.join(prj_dir,pt_file)
model = torch.jit.load(pt_path).to(dev)

raw = torch.tensor(data_raw,dtype=torch.float32).to(dev)

inputs = torch.zeros(1,50,6)
inputs[0] = raw

outputs = model(inputs)

predict = outputs[0].detach().numpy()
print(predict)

x1 = data_raw[:,0]
y1 = data_raw[:,1]
# x2 = label[:,0]
# y2 = label[:,1]
x3 = predict[:,0]
y3 = predict[:,1]

plt.figure(figsize=(10,6))
plt.plot(x1,y1, '*',label='Past Data')
# plt.plot(x2,y2, label='labeled Data')
plt.plot(x3,y3, label='predicted Data')
# plt.axis([-1,1,-1,1])
plt.axis('equal')
plt.legend()
plt.show()       