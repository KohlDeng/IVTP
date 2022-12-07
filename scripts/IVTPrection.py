# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:34:39 2022

@author: DELL
"""

#%% Package import
import numpy as np
import pandas as pd
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data #将数据分批次需要用到它
import torch.nn as nn
import sys
import time
import torch.nn.functional as F
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
matplotlib.rcParams['figure.dpi']= 200



time_begin = time.time()


f = 20 # Hz
past_time = 2.5 # s
pred_time = 5 # s
input_step = round(f*past_time)
output_step = round(f*pred_time)-1

input_size = 6 # 6
output_size = 6

n_epochs = 2 #50
batch_size = 4
#%% Data load

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences): break
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

# 部分HjdriveSet

X = np.array([])
y = np.array([])

for num in range(1):
    filePath = 'E:\\yds\\01_单车模型\\分类数据\\分类数据\\三意图各1000\\'
    folder=os.listdir(filePath)
    for i,item in  enumerate(folder):
       folder[i]=filePath+item
    filename=folder;   
    for item in filename:
        rawdata=pd.read_csv(item, header=None)
        rawdata.columns =["mcutime", "latency", "num_obstacles", "id",
                          "px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration",
                          "heading_rel","cipv_type"]
        label=["px", "py","velocity_abs_x","velocity_abs_y","xAcceleration","yAcceleration"]
        rawdata=rawdata[label]
        X1, y1 = rawdata, rawdata
        
        mm = MinMaxScaler()
        ss = MinMaxScaler()
        X11 = ss.fit_transform(X1)
        y11 = mm.fit_transform(y1)
        
        seq = np.concatenate((X11[np.newaxis,:],y11[np.newaxis,:]), axis = 0)
        if input_step + output_step == seq.shape[1]:
            X_seq = seq[:,:input_step,:]
            y_seq = seq[:,input_step:,:]
        else:
            print("数据维度错误！")
        # X_seq, y_seq = split_sequences(np.array(X11), np.array(y11), input_step, output_step)

        X = np.append(X,X_seq).reshape(-1,input_step,input_size)
        y = np.append(y,y_seq).reshape(-1,output_step,output_size)

# 打乱数据        
index = [i for i in range(X.shape[0])] 
random.shuffle(index)
X = X[index]
y = y[index]

#%% Data preprocess

total_samples = len(X)
cutoff = round(0.2 * total_samples) #训练集、测试集、验证集划分比例：60%、20%、20%

X_train = X[:-cutoff*2]
X_test = X[-cutoff*2:-cutoff]
X_val = X[-cutoff:]

y_train = y[:-cutoff*2]
y_test = y[-cutoff*2:-cutoff]
y_val = y[-cutoff:]

print("Training Shape:", X_train.shape, y_train.shape)
print("Testing Shape:", X_test.shape, y_test.shape)
print("Validating Shape:", X_val.shape, y_val.shape)


if(torch.cuda.is_available()):
    X_train_tensors = torch.autograd.Variable(torch.Tensor(X_train).cuda())    #将数据转为cuda类型
    X_test_tensors = torch.autograd.Variable(torch.Tensor(X_test).cuda())
    X_val_tensors = torch.autograd.Variable(torch.Tensor(X_val).cuda())
    y_train_tensors = torch.autograd.Variable(torch.Tensor(y_train).cuda())
    y_test_tensors = torch.autograd.Variable(torch.Tensor(y_test).cuda())
    y_val_tensors = torch.autograd.Variable(torch.Tensor(y_val).cuda())
else:
    X_train_tensors = torch.autograd.Variable(torch.Tensor(X_train))
    X_test_tensors = torch.autograd.Variable(torch.Tensor(X_test))
    X_val_tensors = torch.autograd.Variable(torch.Tensor(X_val))
    y_train_tensors = torch.autograd.Variable(torch.Tensor(y_train))
    y_test_tensors = torch.autograd.Variable(torch.Tensor(y_test))
    y_val_tensors = torch.autograd.Variable(torch.Tensor(y_val))

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], input_step, X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], input_step, X_test_tensors.shape[2]))
X_val_tensors_final = torch.reshape(X_val_tensors, (X_val_tensors.shape[0], input_step, X_val_tensors.shape[2])) 

torch.manual_seed(1)    # 种子，可复用
BATCH_SIZE = batch_size #设置批次大小

train_data = Data.TensorDataset(X_train_tensors_final, y_train_tensors) #将x,y读取，转换成Tensor格式
test_data = Data.TensorDataset(X_test_tensors_final, y_test_tensors)
val_data = Data.TensorDataset(X_val_tensors_final, y_val_tensors)

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=num_workers)



#%% 模型定义
class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(F.relu(self.linear(x)))
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 output_size = 2,
                 output_step = 50, 
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.output_step = output_step
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell): 
        x = x.unsqueeze(0)
        embedded = self.dropout(F.relu(self.embedding(x)))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, teacher_forcing_ratio = 0):
        batch_size = x.shape[1]
        target_len = y.shape[0]
        
        outputs = torch.zeros(y.shape).to(self.device)
        
        hidden, cell = self.encoder(x)

        decoder_input = x[-1, :, :]
        
        for i in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            outputs[i] = output

            teacher_forcing = random.random() < teacher_forcing_ratio

            decoder_input = y[i] if teacher_forcing else output

        return outputs
    


#%% 模型实例化

INPUT_DIM = input_size
OUTPUT_DIM = output_size
OUTPUT_STEP = output_step
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
learning_rate = 0.0001


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, OUTPUT_STEP, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(enc, dec, dev).to(dev)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#%% Train and Eval

def train(model, dataloader, optimizer, criterion):
    
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(dataloader):       
        # put data into GPU
        x = x.to(dev)
        y = y.to(dev)
        x_trans = x.transpose(0, 1)
        y_trans = y.transpose(0, 1)
        optimizer.zero_grad()
        y_pred = model(x_trans, y_trans, teacher_forcing_ratio = 0.5) 
        loss = criterion(y_pred, y_trans)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(dev)
            y = y.to(dev)
            x_trans = x.transpose(0, 1)
            y_trans = y.transpose(0, 1)
            y_pred = model(x_trans, y_trans, teacher_forcing_ratio = 0)
            loss = criterion(y_pred, y_trans)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
            

N_EPOCHES = n_epochs # 100
best_val_loss = float('inf')

# load previous best model params if exists
model_dir = "saved_models/Seq2Seq"
saved_model_path = model_dir + "/best_seq2seq.pt"
if os.path.isfile(saved_model_path):
    model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
    print("successfully load previous best model parameters")

train_loss_plot = []
val_loss_plot = []  
    
for epoch in range(N_EPOCHES):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_loss_plot = np.append(train_loss_plot, train_loss)
    val_loss_plot = np.append(val_loss_plot, val_loss)
    print(F'Epoch: {epoch+1:02} | Train Loss: {train_loss:.5f} | Val. Loss: {val_loss:.5f}')
    if val_loss < best_val_loss:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), saved_model_path)

plt.plot(train_loss_plot, label='train_loss')
plt.plot(val_loss_plot, label='val_loss')
plt.title('n_epochs='+str(N_EPOCHES))
plt.legend()
plt.savefig("Loss.png", dpi=300)
plt.show()

#%% Test 
results = {'ADE':[], 'FDE':[], 'ADEx':[], 'FDEx':[], 'ADEy':[], 'FDEy':[]}

data_predict = np.array(([[]]*output_step,[[]]*output_step,[[]]*output_step,[[]]*output_step,[[]]*output_step,[[]]*output_step)).T
duration = []
for i in range(X_test_tensors_final.shape[0]):
    X1 = X_test_tensors_final[i].unsqueeze(1)
    y1 = model(X1,torch.zeros([output_step,1,output_size]))
    time_end = time.time()
    duration.append(time_end-time_begin)
    y1 = y1.transpose(0, 1)
    if(torch.cuda.is_available()):
        y1 = y1.cpu().data.numpy()
    else:
        y1 = y1.data.numpy()
    data_predict = np.concatenate((data_predict,y1), axis=0)
dur_average = np.mean(duration) # 单条数据调用模型的平均时长，单位s

if(torch.cuda.is_available()):
    dataY_plot = y_test_tensors.cpu().data.numpy()
    data_past = X_test_tensors_final.cpu().data.numpy()
else:
    dataY_plot = y_test_tensors.data.numpy()
    data_past = X_test_tensors_final.data.numpy()

data_predict = np.array([mm.inverse_transform(data_predict[i,:,:]) for i in range(data_predict.shape[0])])
dataY_plot = np.array([mm.inverse_transform(dataY_plot[i,:,:]) for i in range(dataY_plot.shape[0])])
data_past = np.array([ss.inverse_transform(data_past[i,:,:]) for i in range(data_past.shape[0])])

# 绘图&评价指标计算
for i in range(data_predict.shape[0]):
    x_past = data_past[i][:,0]
    y_past = data_past[i][:,1]
    x_true = dataY_plot[i][:,0]
    y_true = dataY_plot[i][:,1]
    x_predict = data_predict[i][:,0]
    y_predict = data_predict[i][:,1]
    # 评价指标ADE、FDE计算
    true_track = np.transpose(np.array([x_true,y_true]))
    preds_track = np.transpose(np.array([x_predict,y_predict]))
    ADE1 = np.mean ([np.linalg.norm(true_track[x][:2] - preds_track[x][:2]) for x in range(true_track.shape[0])])
    FDE1 = np.linalg.norm(true_track[-1][:2] - preds_track[-1][:2])
    ADE1_x = np.mean ([np.linalg.norm(true_track[x][0] - preds_track[x][0]) for x in range(true_track.shape[0])])
    ADE1_y = np.mean ([np.linalg.norm(true_track[x][1] - preds_track[x][1]) for x in range(true_track.shape[0])])
    FDE1_x = np.linalg.norm(true_track[-1][0] - preds_track[-1][0])
    FDE1_y = np.linalg.norm(true_track[-1][1] - preds_track[-1][1])
    results['ADE'].append(ADE1)
    results['FDE'].append(FDE1)
    results['ADEx'].append(ADE1_x)
    results['FDEx'].append(FDE1_x)
    results['ADEy'].append(ADE1_y)
    results['FDEy'].append(FDE1_y)
    
    # 绘图
    if(i<10):
        plt.figure(figsize=(10,6))
        plt.plot(x_past,y_past, label='Past Data')
        plt.plot(x_true,y_true, label='Actual Data', linestyle='--') # actual plot
        plt.plot(x_predict,y_predict, label='Predicted Data') # predicted plot
        plt.title(F'ADE_y =  {ADE1_y:.2f}m,  FDE_y = {FDE1_y:.2f}m')
        plt.legend()
        plt.savefig('figure_{}.png'.format(i), dpi=300)
        plt.show() 

ADE_ave = np.mean(results['ADE'])
FDE_ave = np.mean(results['FDE'])
ADEx_ave = np.mean(results['ADEx'])
FDEx_ave = np.mean(results['FDEx'])
ADEy_ave = np.mean(results['ADEy'])
FDEy_ave = np.mean(results['FDEy'])
threshold = 5 # m
threshold_x = 3 # m
threshold_y = 3 # m
missing_rate = len([j for j in results['FDE'] if j >threshold])/len(results['FDE'])
missing_rate_x = len([j for j in results['FDEx'] if j >threshold_x])/len(results['FDEx'])
missing_rate_y = len([j for j in results['FDEy'] if j >threshold_y])/len(results['FDEy'])

plt.figure(figsize=(10,6))
plt.scatter([i for i in range(len(results['FDE']))],results['FDE'], s=1, c='orange',label='FDE(m)')
plt.axhline(y=FDE_ave, c='darkorange', linestyle='--', linewidth=2)
plt.scatter([i for i in range(len(results['ADE']))],results['ADE'], s=1, c='g',label='ADE(m)')
plt.axhline(y=ADE_ave, c='darkgreen', linestyle='--', linewidth=2)
plt.ylim([0,max(max(results['ADE']),max(results['FDE']))+1])
plt.title(F'ADE =  {ADE_ave:.2f}m,  FDE = {FDE_ave:.2f}m,  missing rate({threshold}m) = {missing_rate*100:.2f}%')
plt.legend()
plt.savefig('eval_index.png', dpi=300)
plt.show()


#%% 生成C++可调用模型
example_input = torch.ones(1, input_step, input_size).transpose(0, 1).to(dev)
example_output = torch.ones(1, output_step, output_size).transpose(0, 1).to(dev)
traced_script_module = torch.jit.trace(model, (example_input, example_output))
traced_script_module.save("model_1118.pt")

#%%保存整个模型
PATH = model_dir + "/Seq2Seq_model.tar"
torch.save(model, PATH)

time_end = time.time()
duration = time_end - time_begin
print(F"程序运行用时 {int(duration/60)}min{int(duration)%60}s")