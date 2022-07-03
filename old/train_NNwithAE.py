import torch
from torch import nn
import gc

def train(model,optimizer,train_loader,device='cpu',
    loss_reg=nn.MSELoss(),loss_ae=nn.MSELoss(),loss_decoded=nn.L1Loss()
):

    '''
        rmse -> nn.MSELoss
        mae -> nn.L1Loss
    '''

    model.to(device)
    model.train()
    
    running_loss = 0.
    running_reg=0.;running_ae=0;running_decoded=0
    model.train()

    for index, data in enumerate(train_loader):
        X,y = data
        X = X.to(device); y = y.to(device).reshape((-1,1))

        optimizer.zero_grad()
        reg, ae, decoded = model.forward(X)
        loss_reg_ = loss_reg(reg,y)
        loss_ae_ = loss_ae(reg,y)
        loss_decoded_ = loss_decoded(decoded,X)
        loss = loss_reg_ + loss_ae_ + loss_decoded_
        loss.backward()
        optimizer.step()     

        running_loss += loss.item()
        running_reg+=loss_reg_.item();running_ae+=loss_ae_.item()
        running_decoded+=loss_decoded_.item()
        
        del(data)
        gc.collect()
    return running_loss/index, running_reg/index, running_ae/index, running_decoded/index    


@torch.no_grad()
def valid(model,valid_loader,device='cpu',
    loss_reg=nn.MSELoss(),loss_ae=nn.MSELoss(),loss_decoded=nn.L1Loss()):
    
    model.eval()
    running_loss=0;running_reg=0;running_ae=0;running_decoded=0
    for index, data in enumerate(valid_loader):
        X,y = data
        X = X.to(device); y=y.to(device).reshape((-1,1))
        reg,ae,decoded = model.forward(X)
        loss_reg_ = loss_reg(reg,y)
        loss_ae_ = loss_ae(reg,y)
        loss_decoded_ = loss_decoded(decoded,X)
        loss = loss_reg_ + loss_ae_ + loss_decoded_
        running_loss += loss.item()
        running_reg+=loss_reg_.item();running_ae+=loss_ae_.item()
        running_decoded+=loss_decoded_.item()
        
    del(data)
    gc.collect()
        
    return running_loss/index, running_reg/index, running_ae/index, running_decoded/index
