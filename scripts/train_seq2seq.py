# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 2022

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
from seq2seq import Seq2Seq

sample_dir = "/home/kohldyh/Documents/vscode/python/IVTP/dataset/hjdrive_30000"
prj_dir = "/home/kohldyh/Documents/vscode/python/IVTP"

def main():
    dev = 'cuda:0'

    parser = argparse.ArgumentParser(description='PyTorch IVTP example')
    parser.add_argument('--batch_size', type=int, default=500, metavar='BATCH', help='training batch-size (default: 12)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='no. of epochs to run (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MOM', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight',type=str,default="checkpoint/seq_weight.pth.tar",help="weight file name(default:checkpoint/seq_weight.pth.tar)")
        
    hyperparams = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = HjdriveSet(sample_dir,ifscaler=True,scaler_file='config/hjdrive_30000_mminfo.npy')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hyperparams.batch_size, shuffle=True, num_workers=12) 
       
    in_feature=6
    in_steps=50
    out_feature=6
    out_steps=100
    num_layers=1
    
    model = Seq2Seq(in_feature,in_steps,out_feature,out_steps,num_layers,device=dev).to(dev)
    print(model)
    print('dataset size is',trainset.__len__())
    path = os.path.join(prj_dir,hyperparams.weight)
    
    resume = input("Resume training? (Y/N): ")
    if resume == 'Y':
        if os.path.isfile(path):
            train(model,trainloader,hyperparams,dev,path)
        else:
            print("weight file doesn't exit")
    elif resume == 'N':
        train(model,trainloader,hyperparams,dev)
    else:
        print("Invalid input, exiting program.")
        return 0 
    
def save_checkpoint(state,path):
    torch.save(state,path)
    print("check point saved at {}".format(path))  
    
def train(model,trainloader,hyperparams,dev,path=None):
    optimizer = optim.Adam(model.parameters(),lr=hyperparams.lr)
    loss_fn = nn.MSELoss()
    epochs = hyperparams.epochs
    batch_size = hyperparams.batch_size
    if path==None:
        run_epoch = 0   
        list_epoch=[]
        list_loss=[]                      
        
        # 权重初始化似乎作用不大
        for param in model.parameters():
            nn.init.normal_(param, mean=0, std=1)
        
    else:
        print("Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        run_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        list_epoch = checkpoint['list_epoch']
        list_loss = checkpoint['list_loss'] 

    print("start training...")   
    for epoch in range(1+run_epoch,epochs+1+run_epoch):  
        sum_loss = 0.0
        start = time.time()
        count = 0 #for demo test
        for j,data in enumerate(trainloader,1):
            raws,labels = data
            # print(raws)
            # print(labels)
            optimizer.zero_grad()
            output = model(raws.to(dev))
            # print(output)
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
        end = time.time()
        runtime = end-start
        print("epoch {} is over,num_batch is {},runtime is {},loss is {}".format(epoch,num_batch,runtime,avg_loss))
        if(epoch%100==0):    
            print("Saving checkpoint...")
            weight_name = "checkpoint/seq_weight_epoch_"+str(epoch)+".pth.tar"
            weight_path = os.path.join(prj_dir, weight_name)
            save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)

    print("-----------------------------------------------------")
    print("training is over")  
    weight_name = "checkpoint/seq_weight_epoch_"+str(epoch)+".pth.tar"
    weight_path = os.path.join(prj_dir, weight_name)
    save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)

    array_epoch = np.array(list_epoch)
    array_loss = np.array(list_loss)
    plt.figure(1)
    plt.title("Loss During Training")
    plt.plot(array_epoch,array_loss,label="loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    figure_dir = os.path.join(prj_dir,"checkpoint/seq_train_loss.jpg")
    plt.savefig(figure_dir, bbox_inches='tight')
    
if __name__=="__main__":
    main()                      