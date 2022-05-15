

# %%

from imp import is_builtin
import os
import datetime
import sys
import json
from typing import Optional, Union, List
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
from scipy.stats import norm

#%% 

class prepare_dataset_for_train(object) :

    def __init__(self):

        this_file_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        with open(f"{this_file_dir}/local_settings.json","r") as f :
            config_ = json.load(f)

        self.train_files = config_['train_files']
        self.storage = config_['storage']
        self.kaggle_data = config_['kaggle_data']
        self.target = config_['target_dbfs']

    def run(self,
        target_file:str,
        use_feature_files:Union[str,List[str]],
        is_feature_qtrly:Optional[Union[boolean,List[boolean]]]=None,
        normalize_targets:Optional[Union[str,List[str]]]=None,
        normalize_methods:Optional[Union[str,List[str]]]=None
    ):

        target = self.merge(
            target_file,use_feature_files,is_feature_qtrly
        )

        if normalize_targets is None :
            normalize_targets = target.drop(
                columns=['Date','SecuritiesCode','Target']
            )
        if normalize_methods is None :
            normalize_methods = ['blom']*len(normalize_targets)

        target = self.normalize_features(target,
            normalize_target=normalize_targets,
            normalize_method=normalize_methods
        )

        return target


    def merge(
        self,
        target_file:str,
        use_feature_files:Union[str,List[str]],
        is_feature_qtrly:Optional[Union[boolean,List[boolean]]]=None
    ):

        if ~isinstance(use_feature_files,list) :
            use_feature_files = [use_feature_files]
        if is_feature_qtrly is None :
            is_feature_qtrly = [False]*len(use_feature_files)
        if ~isinstance(is_feature_qtrly,list) :
            is_feature_qtrly = [is_feature_qtrly]

        assert len(use_feature_files) == len(is_feature_qtrly), f"""
            length of use_feature_files({len(use_feature_files)}) and 
            lenght of is_feature_qtrly({len(is_feature_qtrly)}) differs
        """

        target = pd.read_csv(target_file)
        target = target.drop_duplicates(subset=['Date','SecuritiesCode'])\
                .reset_index(drop=True)


        for ff,qf in zip(use_feature_files,is_feature_qtrly):

            feat = pd.read_csv(ff)
            if ~qf :
                target = target.merge(feat,on=['Date','SecuritiesCode'],
                    how = "left"
                )
            else :
                target = self.merge_qtrly_feature(target,feat)

        return target

            
    def merge_qtrly_feature(self,target,feat) :

        raise(NotImplementedError('Still Under Construction'))

    def normalize_features(self,
        dataset:pd.DataFrame,
        normalize_target:Union[str,List[str]],
        normalize_method:Union[str,List[str]]="blom"
    ) :

        ds = dataset
        if isinstance(normalize_target,str) :
            normalize_target=[normalize_target]
        nt = normalize_target
        if isinstance(normalize_method,str):
            normalize_method=[normalize_method]*len(nt)
        nm = normalize_method

        dates = ds.Date.unique()

        for tg,mt in zip(nt,mt) :
            for hiduke in dates :
                ds.loc[ds.Date==hiduke,tg] = self._normalize(
                    ds.loc[ds.Date==hiduke,tg].values, mt
                )

        return ds


    def _normalize(self,x:np.ndarray,m:str='blom'):
        if m=='blom':
            x = self._normalize_blom(x)

        return x

    def _normalize_blom(self,x):
        # https://www.statsdirect.com/help/data_preparation/transform_normal_scores.htm
        # filling nan values with mean
        x = np.nan_to_num(x, nan=np.nanmean(x))
        r = np.argsort(np.argsort(x))
        n = len(x)
        return( norm.ppf((r-3/8)/(n+1/4)) )







