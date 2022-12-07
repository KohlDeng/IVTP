# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 2022

@author: kohldyh
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from hjdriveset import HjdriveSet
from mlp import MLPmodel

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/hjdrive_30000"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

def main():
    dev = 'cuda:0'
    
    parser = argparse.ArgumentParser(description='PyTorch IVTP example')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='training batch-size (default: 12)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='E', help='no. of epochs to run (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MOM', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight',type=str,default="checkpoint/mlp_weight.pth.tar",help="weight file name(default:checkpoint/mlp_weight.pth.tar)")
    
    
    hyperparams = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor()])
    
    fullset = HjdriveSet(sample_dir,ifscaler=True,scaler_file='config/hjdrive_30000_mminfo.npy')
    full_size = fullset.__len__()
    train_size = int(0.2*full_size)
    val_size = full_size - train_size
    # torch.manual_seed(0)
    batch_size = hyperparams.batch_size
    trainset,valset = torch.utils.data.random_split(fullset, [train_size,val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    valoader = torch.utils.data.DataLoader(valset,batch_size=hyperparams.batch_size,shuffle=False,num_workers=12)
    
    num_inputs = 50*6
    num_hiddens = 256*6
    num_outputs = 100*6
    
    model = MLPmodel(num_inputs, num_hiddens, num_outputs).to(dev)
    print(model)
    print('trainset size is {}'.format(len(trainloader)*batch_size))
    print('valset size is {}'.format(len(valoader)*batch_size))
    path = os.path.join(prj_dir,hyperparams.weight)
    
    resume = input("Resume training? (Y/N): ")
    
    if resume == 'Y':
        if os.path.isfile(path):
            train(model,trainloader,valoader,hyperparams,dev,path)
        else:
            print("weight file doesn't exit")
    elif resume == 'N':
        train(model,trainloader,valoader,hyperparams,dev)
    else:
        print("Invalid input, exiting program.")
        return 0

    
def save_checkpoint(state,path):
    torch.save(state,path)
    print("check point saved at {}".format(path))    
    
def evaluate_loss(valoader,model,batch_size,loss_fn_val,dev):
    sum_loss = 0.0
    count = 0
    for j,data in enumerate(valoader,1):
        count = count+1
        raws,labels = data
        output = model(raws.to(dev))
        loss = loss_fn_val(output,labels.to(dev))
        sum_loss+=loss.item()
        
    num_batch = count
        
    return sum_loss/num_batch/batch_size

def train(model,trainloader,valoader,hyperparams,dev,path=None):
    optimizer = optim.SGD(model.parameters(),lr=hyperparams.lr,momentum=hyperparams.momentum)
    loss_fn = nn.MSELoss()
    loss_fn_val = nn.MSELoss()
    epochs = hyperparams.epochs
    batch_size = hyperparams.batch_size
    if path==None:
        run_epoch = 0   
        list_epoch=[]
        list_loss=[]
        list_val_loss=[]
        
        # 权重初始化似乎作用不大
        # for param in model.parameters():
        #     nn.init.normal_(param, mean=0, std=0.1)
        
    else:
        print("Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        run_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        list_epoch = checkpoint['list_epoch']
        list_loss = checkpoint['list_loss'] 
        list_val_loss = checkpoint['list_val_loss']
        
    print("start training...")   
    for epoch in range(1+run_epoch,epochs+1+run_epoch):  
        sum_loss = 0.0
        start = time.time()
        count = 0 #for demo test
        for j,data in enumerate(trainloader,1):
            raws,labels = data
            optimizer.zero_grad()
            output = model(raws.to(dev))
            # print(output.shape)
            loss = loss_fn(output,labels.to(dev))
            loss.backward()
            optimizer.step()
            
            sum_loss+=loss.item()
            count+=1
            # if(count>=50):
            #     break
        num_batch = count
        print("-----------------------------------------------------")
        list_epoch.append(epoch)
        avg_loss = sum_loss/batch_size/num_batch
        list_loss.append(avg_loss)
         
        val_loss = evaluate_loss(valoader,model,batch_size,loss_fn_val,dev)
        list_val_loss.append(val_loss)
        end = time.time()
        runtime = end-start

        # val_loss = 0
        print("epoch {} is over,num_batch is {},runtime is {:.2f},loss is {:.4f},val_loss is {:.4f}".format(
            epoch,num_batch,runtime,avg_loss,val_loss))
        
        if(epoch%300==0):    
            print("Saving checkpoint...")
            weight_name = "checkpoint/mlp_weight_epoch_"+str(epoch)+".pth.tar"
            weight_path = os.path.join(prj_dir, weight_name)
            save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'list_val_loss':list_val_loss,
                             'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)
    
    print("-----------------------------------------------------")
    print("training is over")  
    weight_name = "checkpoint/mlp_weight_epoch_"+str(epoch)+".pth.tar"
    weight_path = os.path.join(prj_dir, weight_name)
    save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'list_val_loss':list_val_loss,
                        'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)

    array_epoch = np.array(list_epoch)
    array_loss = np.array(list_loss)
    array_val_loss = np.array(list_val_loss)
    plt.figure(1)
    plt.title("Loss During Training")
    plt.plot(array_epoch,array_loss,label="train_loss")
    plt.plot(array_epoch,array_val_loss,label="val_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    figure_dir = os.path.join(prj_dir,"checkpoint/mlp_train_loss.jpg")
    plt.savefig(figure_dir, bbox_inches='tight')
                    
    
    

if __name__=="__main__":
    main()
    