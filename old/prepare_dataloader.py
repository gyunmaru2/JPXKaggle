# %%
import os, sys, json , datetime, random, gc
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# from torchvision import dataset, transforms


# %%

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self,df, features, labels, transform=None) :
        
#         df = pd.read_csv(file_name)
        self.label_name = labels
        self.feature_name = features
        self.features_values = df[features].values
        self.labels = df[labels].values
        
#         self.file_rows = sum(1 for _ in open(file_name))
    
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx) :
        features_x = torch.tensor(self.features_values[idx]).float()
        labels = torch.tensor(self.labels[idx]).float()
        
        return features_x, labels

class MyDataset_from_file(torch.utils.data.Dataset):

    def __init__(self,file_name,features,labels,transform=None):

        self.file_name = file_name
        self.features = features
        self.labels = labels
        self.transform = transform
        self.file_rows = sum(1 for _ in open(file_name))

    def __len__(self) :
        return self.file_rows - 1 #first row is header

    def __getitem__(self,idx):

        ds = pd.read_csv(self.file_name,
            header=True,
            skiprows=lambda x: x not in idx
        )
        x = torch.tensor(ds.loc[:,self.features].values).float()
        y = torch.tensor(ds.loc[:,self.labels].values).float()

        del ds ; gc.collect()

        return x, y 

# %%
def seed_worker(worker_id=1):
    worker_seed = torch.initial_seed( ) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# %%
def prepare_dataloader_from_file(
    train_file_name:str,
    valid_file_name:str,
    features,label,
    num_workers=1,
    seed=42
):

    assert isinstance(train_file_name,str), """
        file name should be str type
    """
    assert isinstance(valid_file_name,str), """
        file name should be str type
    """

    g = torch.Generator()
    g.manual_seed(seed)

    train_set = MyDataset_from_file(train_file_name,features,label)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set, batch_size=1_024, shuffle = False,
        num_workers = num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    valid_set = MyDataset_from_file(valid_file_name,features,label)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_set, batch_size=1_024, shuffle = False,
        num_workers = num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, valid_loader, train_set, valid_set

# %%
def prepare_dataloader(
    train_file_name:str,
    valid_file_name:str,
    features,label,
    num_workers=1,
    seed=42
):

    assert isinstance(train_file_name,str), """
        file name should be str type
    """
    assert isinstance(valid_file_name,str), """
        file name should be str type
    """

    g = torch.Generator()
    g.manual_seed(seed)

    train_df = pd.read_csv(train_file_name)
    valid_df = pd.read_csv(valid_file_name)

    train_set = MyDataset(train_df,features,label)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set, batch_size=1_024, shuffle = False,
        num_workers = num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    valid_set = MyDataset(valid_df,features,label)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_set, batch_size=1_024, shuffle = False,
        num_workers = num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    del(train_df,valid_df); gc.collect()

    return train_loader, valid_loader, train_set, valid_set
