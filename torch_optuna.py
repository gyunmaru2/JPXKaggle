from multiprocessing.sharedctypes import Value
import torch
from torch import nn
import optuna

from ..model.NNwithAE import NNwithAE
from ..model.early_stopping import EarlyStopping
# from .prepare_dataloader import *
from .train_NNwithAE import *

class optuna_torch(object):
    
    def __init__(self,fixed_params:dict=None,feature_names=None,
        train_loader=None, valid_loader=None, device='cpu'
    ):
        
        self.train_=train; self.valid_=valid
        if fixed_params is None:
            self.fixed_params={
                'num_layer':5,'dropout_rate':.5
            }
        else :
            self.fixed_params = fixed_params

        if feature_names is None :
            raise(ValueError('feature_names is None'))
        else:
            self.feature_names=feature_names

        self.device = device
    
    def objective(self,trial):
        
        feature_names = self.feature_names
        fixed_params = self.fixed_params
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        device = self.device 
        learning_rate = trial.suggest_loguniform('learning_rate',1e-8,1e-1)
        num_layer = fixed_params['num_layer']
        hidden_dim = int(2**trial.suggest_discrete_uniform('hidden_dim',5,10,1))
        dropout_rate = fixed_params['dropout_rate']
        
        model = NNwithAE(
            input_dim=len(feature_names),
            hidden_dim=hidden_dim,dropout_rate=dropout_rate,n_layers=num_layer
        )
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        earlystopping = EarlyStopping(patience=5,verbose=True,path="./tmp.pth")
        
        n_epochs=100
        for epoch in range(n_epochs):
            tl,treg,tae,tdecoded=self.train_(
                model,optimizer,train_loader,device=device)
            vl,vreg,vae,vdecoded=self.valid_(
                model,valid_loader,device=device)
            
            earlystopping(vl,model)
            if earlystopping.early_stop:
                print("early stopping")
                break
                
        del(model)
        gc.collect()
                
        return vl
    
    def run(self, debug_ = False):
        
        n_trials = 100 if not debug_ else 5
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42)
            )
        study.optimize(self.objective,n_trials = n_trials)
        
        self.study = study
        self.best_params = study.best_params
        
        print(self.best_params)
        
        return self.best_params
        