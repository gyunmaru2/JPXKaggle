# %%
import os, sys, json , datetime, random, gc
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import dataset, transforms

# %%

class GaussianNoise(nn.Module):
    
    def __init__(self,p) :
        super(GaussianNoise,self).__init__()
        if p < 0 or p > 1 :
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p 
        self.inplace=False
        
    def forward(self,input):
        return self.gaussian(input,self.training,self.p)
    
    def gaussian(self,ins, is_training, stddev=0.2):
        if is_training:
#             return ins + Variable(torch.randn(ins.size()).to(device) * stddev)
            device = 'cuda' if ins.is_cuda else 'cpu'
            return ins + torch.randn(ins.size()).to(device)*stddev
        return ins
        
# %%

class Encoder(nn.Module):
    
    def __init__(self,d_inp,d_middle,dropout_rate):
        
        super(Encoder,self).__init__()
        
        self.bn1 = nn.BatchNorm1d(d_inp) #バッチ
        self.gn = GaussianNoise(dropout_rate)
        self.fc1 = nn.Linear(d_inp,d_middle)
        self.bn2 = nn.BatchNorm1d(d_middle)
        self.do = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()
        #torch nn.SiLU
        
    def forward(self,inp):
        
        x = self.bn1(inp)
        x = self.gn(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.act(x)
        
        return x

# %%   
class Decoder(nn.Module) :
    
    def __init__(self,d_inp,d_out,dropout_rate):
        
        super(Decoder,self).__init__()
        
        self.do = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_inp,d_out)
        
    def forward(self,inp):
        
        x = self.do(inp)
        x = self.fc1(x)
        
        return x        
    
# %%
class MLP(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,n_layers,dropout_rate) :
        
        assert  n_layers > 2 ,"""
            n_layers shold be larger than 2 
        """
        
        super(MLP,self).__init__()
        
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
         
        self.fcs = nn.ModuleList([nn.Linear(input_dim,hidden_dim)])
        self.dos = nn.ModuleList([nn.Dropout(dropout_rate)])
        
        for _ in range(n_layers-2):
            self.fcs.append(nn.Linear(hidden_dim,hidden_dim))
            self.dos.append(nn.Dropout(dropout_rate))
        self.fc2 = nn.Linear(hidden_dim,1) # regression layer
        
        self.act = nn.SiLU()
        
    def forward(self, inp):
      
        x = self.bn1(inp)
        for fc, do in zip(self.fcs,self.dos):
            x = self.bn2(fc(x))
            x = self.act(x)
            x = do(x)
        x = self.fc2(x)
        
        return x 
        
        
    
# %%       
class NNwithAE(nn.Module):
    
    def __init__(self,input_dim, hidden_dim,dropout_rate,n_layers) :
        
        print(input_dim,hidden_dim,dropout_rate,n_layers)
        
        super(NNwithAE,self).__init__()
        
        self.encoder = Encoder(input_dim,hidden_dim,dropout_rate)
        self.decoder = Decoder(hidden_dim,input_dim,dropout_rate)
        self.mlp = MLP(input_dim+hidden_dim,hidden_dim,n_layers,dropout_rate)
        
        # regression or classificatio for aute encoder
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.do = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim,1)
        
        self.bn2 = nn.BatchNorm1d(input_dim)
        
        self.act = nn.SiLU()
        
        
    def forward(self,inpt) :
        
        encoded = self.encoder(inpt)
        decoded = self.decoder(encoded)
        
        x_ae = self.fc1(decoded)
        x_ae = self.do(self.act(self.bn(x_ae)))
        x_ae = self.fc2(x_ae)
        
        x0 = self.bn2(inpt)
        x_reg = torch.cat([x0,encoded],dim=1)
        
        x_reg = self.mlp(x_reg)
        
        return x_reg,x_ae,decoded
        
        
        