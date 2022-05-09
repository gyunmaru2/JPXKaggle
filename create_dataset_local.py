import numpy as np
import pandas as pd
import shutil
import datetime
import os
import sys
import json

from tech_etl import technicals_etl
from fin_etl import financials_etl
from create_dataset_kaggle import prepare_dataset

class prepare_dataset_local(prepare_dataset):

    files = ['stock_prices','financials','secondary_stock_prices',
        'options','trades']

    def __init__(self,version=None):

        with open("./local_settings.json","r") as f:
            config_= json.load(f)

        self.storage_dir = config_['storage']
        self.train_files = config_['train_files']

        if version is None :
            version = datetime.datetime.today()\
                    .strftime("%Y%m%d%H%M%S")
        if version[-1] != "/" :
            self.storage_dir += version + "/"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        for file in self.files :

            if os.path.exists(self.storage_dir+file+".csv") :
                os.remove(self.storage_dir+file+".csv")

            shutil.copyfile(self.train_files+file+".csv",
                self.storage_dir+file+".csv"
            )

        